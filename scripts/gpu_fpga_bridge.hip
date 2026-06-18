/*
 * gpu_fpga_bridge.hip — Persistent GPU<->FPGA bridge
 * ===================================================
 * Single binary: HIP kernel + host UDP bridge loop.
 *
 * GPU kernel: 64 LIF neurons, persistent loop, reads/writes shared memory.
 * Host thread: UDP loop at ~500-1000Hz, sends GPU states to FPGA, receives
 *              FPGA telemetry, writes back to shared memory.
 *
 * Build: hipcc --offload-arch=gfx1100 -O2 -o scripts/gpu_fpga_bridge scripts/gpu_fpga_bridge.hip -lpthread
 * Run:   HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/gpu_fpga_bridge [mode] [rate_hz] [duration_s]
 *
 * Requires: FPGA at 192.168.0.50:7700 with ETH bridge bitstream
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <pthread.h>
#include <unistd.h>
#include <signal.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

// ---- Constants ----
#define GPU_NEURONS    64
#define FPGA_NEURONS   128
#define TAU_FAST       0.90f
#define TAU_SLOW       0.99f
#define THRESHOLD      1.0f

// FPGA protocol
#define FPGA_IP        "192.168.0.50"
#define FPGA_PORT      7700
#define LOCAL_PORT     7701
#define CMD_SET_VG       0x01
#define CMD_READ_TELEM   0x02
#define CMD_SET_KILL     0x03
#define CMD_SET_MAC      0x06
#define CMD_SET_VG_BATCH 0x08
#define CMD_SET_RATE     0x09
#define TELEM_PACKET_LEN 1029
#define TELEM_PAYLOAD_LEN 1024

// ---- Shared Memory Layout (host-mapped, GPU-visible) ----
struct SharedState {
    // GPU -> Host (neuron states)
    float gpu_vmem[GPU_NEURONS];
    float gpu_spikes[GPU_NEURONS];
    float gpu_adapt[GPU_NEURONS];
    float gpu_spike_rate[GPU_NEURONS];

    // Host -> GPU (FPGA feedback + input)
    float fpga_spikes[FPGA_NEURONS];
    float fpga_vmem[FPGA_NEURONS];
    float input_signal;
    int   mode;               // 0=standard, 1=adaptation, 2=recurrence, 3=full

    // Sync
    volatile int gpu_step;
    volatile int host_step;
    volatile int running;

    // Stats (written by host)
    float loop_rate_hz;
    float round_trip_us;
    int   total_gpu_steps;
    int   total_host_steps;
    int   fpga_timeouts;

    // Input weight seed (set once by host)
    float w_in[GPU_NEURONS];
};

// ---- Persistent GPU Kernel ----
// One thread per neuron, runs in a loop until shared->running == 0
__global__ void persistent_lif_kernel(SharedState* shared) {
    int nid = threadIdx.x;
    if (nid >= GPU_NEURONS) return;

    float v = 0.0f;
    float w = 0.0f;  // adaptation current
    float spike_ema = 0.0f;
    unsigned long long t0 = clock64();
    int my_step = 0;
    int mode = shared->mode;

    while (shared->running) {
        // Wait for host to advance
        // Spin-wait with memory fence
        int target = my_step + 1;
        int spin = 0;
        while (shared->host_step < target) {
            if (!shared->running) return;
            // Yield after some spins to avoid burning GPU
            if (++spin > 1000) {
                // Brief pause — use clock64 busy-wait (~1us)
                unsigned long long wait_start = clock64();
                while (clock64() - wait_start < 1000) {}
                spin = 0;
            }
            __threadfence();
        }

        mode = shared->mode;

        // --- Substrate noise from hardware clock ---
        unsigned long long tnow = clock64();
        float noise = (float)((tnow ^ (t0 + my_step * 7919ULL + nid * 131ULL)) % 10000) / 50000.0f;

        // --- External input ---
        float I_ext = shared->input_signal * shared->w_in[nid];

        // --- FPGA feedback coupling ---
        // Mix in FPGA spike data as additional input
        float I_fpga = 0.0f;
        {
            // Each GPU neuron couples to 2 FPGA neurons (interleaved)
            int fpga_a = nid * 2;
            int fpga_b = nid * 2 + 1;
            if (fpga_a < FPGA_NEURONS)
                I_fpga += 0.02f * shared->fpga_spikes[fpga_a];
            if (fpga_b < FPGA_NEURONS)
                I_fpga += 0.01f * shared->fpga_spikes[fpga_b];
        }

        // --- Recurrent input via cross-lane shuffle ---
        float I_rec = 0.0f;
        if (mode >= 2) {
            float my_spike = shared->gpu_spikes[nid];
            float s1 = __shfl_xor(my_spike, 1);
            float s2 = __shfl_xor(my_spike, 2);
            float s4 = __shfl_xor(my_spike, 4);
            I_rec = 0.1f * s1 + 0.05f * s2 + 0.02f * s4;
        }

        // --- Adaptation current ---
        float I_adapt = 0.0f;
        if (mode >= 1) {
            I_adapt = -0.1f * w;
            w = TAU_SLOW * w;
        }

        // --- Membrane dynamics ---
        v = TAU_FAST * v + I_ext + noise + I_rec + I_adapt + I_fpga;

        // --- Spike check ---
        float spiked = 0.0f;
        if (v >= THRESHOLD) {
            spiked = 1.0f;
            v = 0.0f;
            if (mode >= 1) w += 0.5f;
        }

        // --- Write outputs ---
        shared->gpu_vmem[nid] = v;
        shared->gpu_spikes[nid] = spiked;
        if (mode >= 1) shared->gpu_adapt[nid] = w;

        // EMA spike rate (alpha = 0.01)
        spike_ema = 0.99f * spike_ema + 0.01f * spiked;
        shared->gpu_spike_rate[nid] = spike_ema;

        __threadfence();

        // Only thread 0 increments gpu_step
        if (nid == 0) {
            atomicAdd((int*)&shared->gpu_step, 1);
        }
        __syncthreads();

        my_step++;
    }
}

// ---- CRC-8 (polynomial 0x07, matching FPGA firmware) ----
static uint8_t crc8(const uint8_t* data, int len) {
    uint8_t crc = 0;
    for (int i = 0; i < len; i++) {
        crc ^= data[i];
        for (int b = 0; b < 8; b++) {
            if (crc & 0x80)
                crc = (uint8_t)((crc << 1) ^ 0x07);
            else
                crc = (uint8_t)(crc << 1);
        }
    }
    return crc;
}

// Forward declarations
static void write_ts_record(SharedState* shared, int step);

// ---- Timing helpers ----
static double now_us() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

// ---- Global state for signal handler ----
static SharedState* g_shared = nullptr;
static void signal_handler(int sig) {
    (void)sig;
    if (g_shared) g_shared->running = 0;
    fprintf(stderr, "\n[BRIDGE] Signal caught, shutting down...\n");
}

// ---- UDP Bridge Host Loop ----
struct BridgeArgs {
    SharedState* shared;
    int target_rate_hz;
    int udp_sock;
    struct sockaddr_in fpga_addr;
};

static void* bridge_loop(void* arg) {
    BridgeArgs* ba = (BridgeArgs*)arg;
    SharedState* shared = ba->shared;
    int sock = ba->udp_sock;
    struct sockaddr_in fpga_addr = ba->fpga_addr;

    int target_rate = ba->target_rate_hz;
    int period_us = (target_rate > 0) ? (1000000 / target_rate) : 2000;  // default 500 Hz

    uint8_t tx_buf[1024];
    uint8_t rx_buf[2048];
    int host_step = 0;

    double t_start = now_us();
    double t_last_stat = t_start;
    int loops_since_stat = 0;
    int timeouts = 0;

    // Set socket timeout to 5ms for non-blocking-ish receive
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 5000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    fprintf(stderr, "[BRIDGE] UDP loop started, target %d Hz (period %d us)\n",
            target_rate, period_us);

    while (shared->running) {
        double t0 = now_us();

        // 1. Read GPU neuron states and compute MAC signal
        float mac = 0.0f;
        for (int i = 0; i < GPU_NEURONS; i++) {
            mac += shared->gpu_spike_rate[i];
        }
        mac /= GPU_NEURONS;  // average spike rate [0, 1]

        // 2. Send MAC + VG modulation to FPGA
        // MAC: average GPU spike rate
        {
            uint32_t mac_q16 = (uint32_t)(mac * 65536.0f) & 0xFFFFFFFF;
            tx_buf[0] = 0x55;
            tx_buf[1] = CMD_SET_MAC;
            tx_buf[2] = (mac_q16 >> 24) & 0xFF;
            tx_buf[3] = (mac_q16 >> 16) & 0xFF;
            tx_buf[4] = (mac_q16 >>  8) & 0xFF;
            tx_buf[5] = (mac_q16      ) & 0xFF;
            sendto(sock, tx_buf, 6, 0, (struct sockaddr*)&fpga_addr, sizeof(fpga_addr));
        }

        // 2b. Modulate FPGA gate voltages based on input + GPU states
        // Only every 10 steps (~50Hz) to avoid flooding FPGA UDP parser
        if (host_step % 10 == 0) {
            float input_val = shared->input_signal;
            float alpha = 0.25f;  // input coupling strength
            float beta = 0.08f;   // per-neuron GPU coupling

            tx_buf[0] = 0x55;
            tx_buf[1] = CMD_SET_VG_BATCH;
            tx_buf[2] = 0;               // start_id = 0
            tx_buf[3] = FPGA_NEURONS;    // count = 128
            for (int n = 0; n < FPGA_NEURONS; n++) {
                // Heterogeneous base Vg
                float base_vg = 0.505f + 0.15f * ((float)(n % 17) / 16.0f);
                // Input modulation
                float vg = base_vg + alpha * (input_val - 0.5f);
                // GPU neuron coupling (each FPGA neuron couples to a GPU neuron)
                int gpu_src = n % GPU_NEURONS;
                vg += beta * shared->gpu_spike_rate[gpu_src];
                // Clamp to safe range
                if (vg < 0.0f) vg = 0.0f;
                if (vg > 1.0f) vg = 1.0f;

                uint32_t vg_q16 = (uint32_t)(vg * 65536.0f) & 0xFFFFFFFF;
                int off = 4 + n * 4;  // after header(2) + start_id(1) + count(1)
                tx_buf[off + 0] = (vg_q16 >> 24) & 0xFF;
                tx_buf[off + 1] = (vg_q16 >> 16) & 0xFF;
                tx_buf[off + 2] = (vg_q16 >>  8) & 0xFF;
                tx_buf[off + 3] = (vg_q16      ) & 0xFF;
            }
            sendto(sock, tx_buf, 4 + FPGA_NEURONS * 4, 0,
                   (struct sockaddr*)&fpga_addr, sizeof(fpga_addr));
        }

        // 3. Request telemetry: [0x55][CMD_READ_TELEM]
        {
            tx_buf[0] = 0x55;
            tx_buf[1] = CMD_READ_TELEM;
            sendto(sock, tx_buf, 2, 0, (struct sockaddr*)&fpga_addr, sizeof(fpga_addr));
        }

        // 4. Receive FPGA telemetry
        {
            struct sockaddr_in src_addr;
            socklen_t src_len = sizeof(src_addr);
            ssize_t n = recvfrom(sock, rx_buf, sizeof(rx_buf), 0,
                                 (struct sockaddr*)&src_addr, &src_len);
            if (n >= TELEM_PACKET_LEN) {
                // Validate header
                if (rx_buf[0] == 0x55 && rx_buf[1] == 0x02) {
                    int payload_len = (rx_buf[2] << 8) | rx_buf[3];
                    if (payload_len == TELEM_PAYLOAD_LEN) {
                        // Validate CRC
                        uint8_t expected_crc = crc8(rx_buf, TELEM_PACKET_LEN - 1);
                        if (rx_buf[TELEM_PACKET_LEN - 1] == expected_crc) {
                            // Parse 128 neurons: each 8 bytes
                            // spike_count(2) + vmem_q16(4) + refract(2)
                            for (int i = 0; i < FPGA_NEURONS; i++) {
                                int off = 4 + i * 8;
                                uint16_t sc = (rx_buf[off] << 8) | rx_buf[off+1];
                                int32_t vm_raw = (rx_buf[off+2] << 24) | (rx_buf[off+3] << 16) |
                                                 (rx_buf[off+4] << 8)  | rx_buf[off+5];
                                shared->fpga_spikes[i] = (float)sc;  // raw count
                                shared->fpga_vmem[i] = (float)vm_raw / 65536.0f;
                            }
                        }
                    }
                }
            } else if (n < 0) {
                timeouts++;
            }
        }

        // 5. Write timeseries record (before advancing step)
        write_ts_record(shared, host_step);

        // 5b. Advance host step (unblocks GPU kernel)
        host_step++;
        shared->host_step = host_step;
        shared->total_host_steps = host_step;
        shared->fpga_timeouts = timeouts;
        __sync_synchronize();  // memory barrier

        // 6. Rate control
        double elapsed = now_us() - t0;
        double remaining = (double)period_us - elapsed;
        if (remaining > 0) {
            usleep((useconds_t)remaining);
        }

        // 7. Stats every second
        loops_since_stat++;
        double t_now = now_us();
        if (t_now - t_last_stat >= 1e6) {
            double actual_rate = loops_since_stat * 1e6 / (t_now - t_last_stat);
            shared->loop_rate_hz = (float)actual_rate;
            shared->round_trip_us = (float)elapsed;
            fprintf(stderr, "[BRIDGE] rate=%.1f Hz  rtt=%.0f us  gpu_step=%d  timeouts=%d\n",
                    actual_rate, elapsed, shared->gpu_step, timeouts);
            loops_since_stat = 0;
            t_last_stat = t_now;
        }
    }

    fprintf(stderr, "[BRIDGE] UDP loop stopped after %d steps\n", host_step);
    return NULL;
}

// ---- Input generator thread (generates test signals) ----
struct InputGenArgs {
    SharedState* shared;
    int pattern;  // 0=sine, 1=square, 2=noise, 3=ramp
    float freq;   // Hz
};

static void* input_gen_loop(void* arg) {
    InputGenArgs* ig = (InputGenArgs*)arg;
    SharedState* shared = ig->shared;
    int step = 0;

    while (shared->running) {
        float t = (float)step * 0.001f;  // 1ms per step
        float val = 0.0f;

        switch (ig->pattern) {
            case 0: val = 0.5f * sinf(2.0f * M_PI * ig->freq * t) + 0.5f; break;
            case 1: val = (fmodf(t * ig->freq, 1.0f) > 0.5f) ? 1.0f : 0.0f; break;
            case 2: val = (float)(rand() % 10000) / 10000.0f; break;
            case 3: val = fmodf(t * ig->freq, 1.0f); break;
        }

        shared->input_signal = val;
        step++;
        usleep(1000);  // 1kHz input update
    }
    return NULL;
}

// ---- Timeseries dump file ----
// Binary format: each record = [step(int32), input(float), gpu_vmem(64*float), gpu_spikes(64*float), fpga_spikes(128*float), fpga_vmem(128*float)]
// Record size = 4 + 4 + 64*4 + 64*4 + 128*4 + 128*4 = 1544 bytes
#define TS_RECORD_SIZE (4 + 4 + GPU_NEURONS*4 + GPU_NEURONS*4 + FPGA_NEURONS*4 + FPGA_NEURONS*4)

static FILE* g_ts_file = nullptr;

static void write_ts_record(SharedState* shared, int step) {
    if (!g_ts_file) return;
    fwrite(&step, 4, 1, g_ts_file);
    float inp = shared->input_signal;
    fwrite(&inp, 4, 1, g_ts_file);
    fwrite(shared->gpu_vmem, 4, GPU_NEURONS, g_ts_file);
    fwrite(shared->gpu_spikes, 4, GPU_NEURONS, g_ts_file);
    fwrite(shared->fpga_spikes, 4, FPGA_NEURONS, g_ts_file);
    fwrite(shared->fpga_vmem, 4, FPGA_NEURONS, g_ts_file);
}

// ---- JSON output for Python benchmark ----
static void print_json_stats(SharedState* shared, double total_time_s) {
    printf("{\n");
    printf("  \"total_time_s\": %.3f,\n", total_time_s);
    printf("  \"gpu_steps\": %d,\n", shared->total_gpu_steps);
    printf("  \"host_steps\": %d,\n", shared->total_host_steps);
    printf("  \"loop_rate_hz\": %.1f,\n", shared->loop_rate_hz);
    printf("  \"round_trip_us\": %.1f,\n", shared->round_trip_us);
    printf("  \"fpga_timeouts\": %d,\n", shared->fpga_timeouts);

    // GPU neuron stats
    printf("  \"gpu_vmem\": [");
    for (int i = 0; i < GPU_NEURONS; i++) {
        printf("%.4f%s", shared->gpu_vmem[i], (i < GPU_NEURONS-1) ? ", " : "");
    }
    printf("],\n");

    printf("  \"gpu_spike_rate\": [");
    for (int i = 0; i < GPU_NEURONS; i++) {
        printf("%.6f%s", shared->gpu_spike_rate[i], (i < GPU_NEURONS-1) ? ", " : "");
    }
    printf("],\n");

    // FPGA neuron stats (first 16)
    printf("  \"fpga_spikes_sample\": [");
    for (int i = 0; i < 16; i++) {
        printf("%.0f%s", shared->fpga_spikes[i], (i < 15) ? ", " : "");
    }
    printf("],\n");

    printf("  \"fpga_vmem_sample\": [");
    for (int i = 0; i < 16; i++) {
        printf("%.4f%s", shared->fpga_vmem[i], (i < 15) ? ", " : "");
    }
    printf("]\n");

    printf("}\n");
}

// ---- Main ----
int main(int argc, char** argv) {
    int mode = 3;             // default: full mode
    int rate_hz = 500;        // default: 500 Hz bridge loop
    float duration_s = 10.0f; // default: 10 seconds
    int input_pattern = 0;    // default: sine
    float input_freq = 2.0f;  // default: 2 Hz
    int json_output = 0;      // if 1, output JSON to stdout at end
    const char* ts_path = nullptr;  // timeseries dump file path

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i+1 < argc) mode = atoi(argv[++i]);
        else if (strcmp(argv[i], "--rate") == 0 && i+1 < argc) rate_hz = atoi(argv[++i]);
        else if (strcmp(argv[i], "--duration") == 0 && i+1 < argc) duration_s = atof(argv[++i]);
        else if (strcmp(argv[i], "--pattern") == 0 && i+1 < argc) input_pattern = atoi(argv[++i]);
        else if (strcmp(argv[i], "--freq") == 0 && i+1 < argc) input_freq = atof(argv[++i]);
        else if (strcmp(argv[i], "--json") == 0) json_output = 1;
        else if (strcmp(argv[i], "--timeseries") == 0 && i+1 < argc) ts_path = argv[++i];
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr, "Usage: %s [options]\n", argv[0]);
            fprintf(stderr, "  --mode N       LIF mode: 0=standard, 1=adapt, 2=recur, 3=full (default: 3)\n");
            fprintf(stderr, "  --rate Hz      Bridge loop rate in Hz (default: 500)\n");
            fprintf(stderr, "  --duration S   Run duration in seconds (default: 10)\n");
            fprintf(stderr, "  --pattern N    Input: 0=sine, 1=square, 2=noise, 3=ramp (default: 0)\n");
            fprintf(stderr, "  --freq Hz      Input signal frequency (default: 2.0)\n");
            fprintf(stderr, "  --json         Output JSON stats to stdout at end\n");
            fprintf(stderr, "  --timeseries F Write per-step binary timeseries to file F\n");
            return 0;
        }
    }

    fprintf(stderr, "[BRIDGE] GPU-FPGA Persistent Bridge\n");
    fprintf(stderr, "[BRIDGE] Mode=%d  Rate=%d Hz  Duration=%.1f s  Pattern=%d  Freq=%.1f Hz\n",
            mode, rate_hz, duration_s, input_pattern, input_freq);

    // ---- Allocate host-mapped memory ----
    SharedState* shared = nullptr;
    hipError_t err = hipHostMalloc(&shared, sizeof(SharedState),
                                   hipHostMallocMapped | hipHostMallocWriteCombined);
    if (err != hipSuccess) {
        // Fallback without WriteCombined
        err = hipHostMalloc(&shared, sizeof(SharedState), hipHostMallocMapped);
    }
    if (err != hipSuccess) {
        fprintf(stderr, "[BRIDGE] FATAL: hipHostMalloc failed: %s\n", hipGetErrorString(err));
        return 1;
    }

    memset(shared, 0, sizeof(SharedState));
    shared->mode = mode;
    shared->running = 1;

    // Initialize input weights (deterministic)
    srand(42);
    for (int i = 0; i < GPU_NEURONS; i++) {
        shared->w_in[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    // Get device pointer for the mapped memory
    SharedState* d_shared = nullptr;
    err = hipHostGetDevicePointer((void**)&d_shared, shared, 0);
    if (err != hipSuccess) {
        fprintf(stderr, "[BRIDGE] FATAL: hipHostGetDevicePointer failed: %s\n", hipGetErrorString(err));
        hipHostFree(shared);
        return 1;
    }

    g_shared = shared;

    // ---- Setup UDP socket ----
    int udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_sock < 0) {
        perror("[BRIDGE] socket");
        hipHostFree(shared);
        return 1;
    }

    int reuse = 1;
    setsockopt(udp_sock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    struct sockaddr_in local_addr;
    memset(&local_addr, 0, sizeof(local_addr));
    local_addr.sin_family = AF_INET;
    local_addr.sin_port = htons(LOCAL_PORT);
    local_addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(udp_sock, (struct sockaddr*)&local_addr, sizeof(local_addr)) < 0) {
        perror("[BRIDGE] bind");
        close(udp_sock);
        hipHostFree(shared);
        return 1;
    }

    struct sockaddr_in fpga_addr;
    memset(&fpga_addr, 0, sizeof(fpga_addr));
    fpga_addr.sin_family = AF_INET;
    fpga_addr.sin_port = htons(FPGA_PORT);
    inet_pton(AF_INET, FPGA_IP, &fpga_addr.sin_addr);

    // Send kill-switch-off
    {
        uint8_t pkt[] = {0x55, CMD_SET_KILL, 0x00};
        sendto(udp_sock, pkt, 3, 0, (struct sockaddr*)&fpga_addr, sizeof(fpga_addr));
        usleep(10000);
    }

    fprintf(stderr, "[BRIDGE] UDP socket ready, FPGA=%s:%d, local port=%d\n",
            FPGA_IP, FPGA_PORT, LOCAL_PORT);

    // Open timeseries dump file if requested
    if (ts_path) {
        g_ts_file = fopen(ts_path, "wb");
        if (!g_ts_file) {
            fprintf(stderr, "[BRIDGE] WARNING: Could not open timeseries file: %s\n", ts_path);
        } else {
            // Write header: magic(4) + gpu_neurons(int32) + fpga_neurons(int32) + record_size(int32)
            int32_t hdr[4] = {0x54535246, GPU_NEURONS, FPGA_NEURONS, TS_RECORD_SIZE};  // "FRST" magic
            fwrite(hdr, 4, 4, g_ts_file);
            fprintf(stderr, "[BRIDGE] Timeseries dump: %s (record=%d bytes)\n", ts_path, TS_RECORD_SIZE);
        }
    }

    // ---- Install signal handler ----
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // ---- Launch GPU kernel ----
    fprintf(stderr, "[BRIDGE] Launching persistent GPU kernel with %d neurons...\n", GPU_NEURONS);
    hipStream_t stream;
    hipStreamCreate(&stream);
    persistent_lif_kernel<<<1, GPU_NEURONS, 0, stream>>>(d_shared);

    // ---- Start bridge thread ----
    BridgeArgs bridge_args;
    bridge_args.shared = shared;
    bridge_args.target_rate_hz = rate_hz;
    bridge_args.udp_sock = udp_sock;
    bridge_args.fpga_addr = fpga_addr;

    pthread_t bridge_thread;
    pthread_create(&bridge_thread, NULL, bridge_loop, &bridge_args);

    // ---- Start input generator thread ----
    InputGenArgs input_args;
    input_args.shared = shared;
    input_args.pattern = input_pattern;
    input_args.freq = input_freq;

    pthread_t input_thread;
    pthread_create(&input_thread, NULL, input_gen_loop, &input_args);

    // ---- Wait for duration ----
    double start_time = now_us();
    while (shared->running) {
        usleep(100000);  // check every 100ms
        double elapsed = (now_us() - start_time) / 1e6;
        if (elapsed >= duration_s) {
            shared->running = 0;
        }
    }

    // ---- Shutdown ----
    fprintf(stderr, "[BRIDGE] Stopping...\n");
    shared->running = 0;
    __sync_synchronize();

    // Advance host_step one more time to unblock any waiting GPU threads
    shared->host_step = shared->total_host_steps + 10;
    __sync_synchronize();

    pthread_join(input_thread, NULL);
    pthread_join(bridge_thread, NULL);

    // Wait for GPU kernel to finish
    hipStreamSynchronize(stream);
    hipStreamDestroy(stream);

    double total_time = (now_us() - start_time) / 1e6;

    fprintf(stderr, "[BRIDGE] Done. GPU steps: %d, Host steps: %d, Time: %.1f s\n",
            shared->gpu_step, shared->total_host_steps, total_time);
    fprintf(stderr, "[BRIDGE] Final rate: %.1f Hz, RTT: %.0f us, Timeouts: %d\n",
            shared->loop_rate_hz, shared->round_trip_us, shared->fpga_timeouts);

    // ---- Output ----
    if (json_output) {
        shared->total_gpu_steps = shared->gpu_step;
        print_json_stats(shared, total_time);
    }

    // ---- Cleanup ----
    if (g_ts_file) {
        fclose(g_ts_file);
        fprintf(stderr, "[BRIDGE] Timeseries file closed (%d records)\n", shared->total_host_steps);
    }
    close(udp_sock);
    hipHostFree(shared);

    return 0;
}
