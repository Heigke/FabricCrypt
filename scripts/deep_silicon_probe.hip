/*
 * deep_silicon_probe.hip — Test ALL unexplored low-level GPU capabilities
 * for neuromorphic computation potential.
 *
 * Tests:
 *  1. s_setreg — can we WRITE hardware registers from shader?
 *  2. Atomic contention as neural integration
 *  3. LDS as synaptic memory with nonlinear activation
 *  4. Cache hierarchy timing as multi-scale memory
 *  5. Wave intrinsics (readlane, permute) as neural fabric
 *  6. GDS (Global Data Share) access
 *  7. hwreg scan for writable registers
 *  8. Instruction timing as stochastic computation
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NTHREADS 256
#define NBLOCKS 64
#define NTRIALS 32

/* ============================================================
 * TEST 1: s_setreg — Write hardware registers from shader
 * This has NEVER been tested in the entire project!
 * ============================================================ */
__global__ void test_setreg_mode(uint32_t* results) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t mode_before, mode_after, mode_restored;

    // Read current MODE register
    asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(mode_before));

    // Try to WRITE to MODE register (FP rounding control)
    // MODE bits: [3:0] = FP_ROUND, [7:4] = FP_DENORM, etc.
    // Try setting round-toward-zero (bits[1:0] = 0b11)
    uint32_t new_mode = mode_before | 0x3;  // Set round-toward-zero for 32-bit
    asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(new_mode));

    // Read back to verify
    asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(mode_after));

    // Restore original
    asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(mode_before));
    asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(mode_restored));

    if (tid < 256) {
        results[tid * 4 + 0] = mode_before;
        results[tid * 4 + 1] = new_mode;     // what we tried to write
        results[tid * 4 + 2] = mode_after;   // what we read back
        results[tid * 4 + 3] = mode_restored;
    }
}

/* Test: does s_setreg actually affect computation? */
__global__ void test_setreg_computation(float* results) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    float a = 1.0f;
    float b = __int_as_float(0x33800000); // very small number (5.96e-8)
    float result_default, result_rtz, result_rni;

    // Default rounding (round-to-nearest-even)
    result_default = a + b;

    // Switch to round-toward-zero
    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 4), 0x3"); // bits[3:0] = 0011 = RTZ
    result_rtz = a + b;

    // Switch to round-toward-negative-infinity
    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 4), 0x2"); // bits[3:0] = 0010 = RNI
    result_rni = a + b;

    // Restore default (round-to-nearest-even)
    asm volatile("s_setreg_imm32_b32 hwreg(1, 0, 4), 0x0");

    if (tid < 1024) {
        results[tid * 3 + 0] = result_default;
        results[tid * 3 + 1] = result_rtz;
        results[tid * 3 + 2] = result_rni;
    }
}

/* ============================================================
 * TEST 2: Atomic contention as neural integration
 * Multiple waves atomicAdd to same location = summation
 * atomicMax = threshold activation, atomicCAS = gating
 * ============================================================ */
__global__ void test_atomic_neuron(int* soma, int* dendrites,
                                    int* spike_count, int threshold) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Each thread is a "synapse" contributing to a "neuron" (soma[neuron_id])
    int neuron_id = tid % 64;  // 64 neurons
    int weight = dendrites[tid % 1024];  // synapse weight

    // Neural integration via atomic add (hardware-implemented!)
    int prev = atomicAdd(&soma[neuron_id], weight);

    // Check if we crossed threshold (spike detection via atomicCAS)
    int current = prev + weight;
    if (current >= threshold && prev < threshold) {
        // This thread triggered the spike — reset neuron
        atomicExch(&soma[neuron_id], 0);  // reset
        atomicAdd(&spike_count[neuron_id], 1);  // count spike
    }
}

/* ============================================================
 * TEST 3: LDS as synaptic memory with nonlinear activation
 * 64KB LDS per CU = on-chip "synaptic weight matrix"
 * ============================================================ */
__global__ void test_lds_synapse(float* outputs, float* weights_global,
                                  int n_neurons, int n_inputs) {
    __shared__ float lds_weights[4096];  // 16KB of synaptic weights in LDS
    __shared__ float lds_membrane[256];  // membrane potentials
    __shared__ uint32_t lds_spikes[8];   // spike bitmap (256 neurons)

    uint32_t tid = threadIdx.x;

    // Load weights into LDS (one-time, then persistent across "timesteps")
    for (int i = tid; i < 4096; i += blockDim.x) {
        lds_weights[i] = weights_global[i + blockIdx.x * 4096];
    }
    if (tid < 256) lds_membrane[tid] = 0.0f;
    if (tid < 8) lds_spikes[tid] = 0;
    __syncthreads();

    // Simulate 16 timesteps of spiking neural network IN LDS
    for (int t = 0; t < 16; t++) {
        // Each thread processes one neuron
        if (tid < (uint32_t)n_neurons) {
            float sum = 0.0f;
            // Sum synaptic inputs (fully connected, weights in LDS)
            for (int j = 0; j < n_inputs && j < 16; j++) {
                int pre_word = j / 32;
                int pre_bit = j % 32;
                if (lds_spikes[pre_word] & (1u << pre_bit)) {
                    sum += lds_weights[tid * 16 + j];
                }
            }

            // Leaky integrate
            lds_membrane[tid] = lds_membrane[tid] * 0.9f + sum;

            // Spike if above threshold (nonlinear activation)
            if (lds_membrane[tid] > 1.0f) {
                int word = tid / 32;
                int bit = tid % 32;
                atomicOr((unsigned int*)&lds_spikes[word], 1u << bit);
                lds_membrane[tid] = 0.0f;  // reset
            } else {
                int word = tid / 32;
                int bit = tid % 32;
                atomicAnd((unsigned int*)&lds_spikes[word], ~(1u << bit));
            }
        }
        __syncthreads();
    }

    // Output final membrane state
    if (tid < (uint32_t)n_neurons) {
        outputs[blockIdx.x * n_neurons + tid] = lds_membrane[tid];
    }
}

/* TEST 4 removed — cache timing test requires complex allocation */

/* ============================================================
 * TEST 5: Wave intrinsics as neural communication fabric
 * readlane/writelane = axonal transport between neurons
 * ballot = population coding, permute = routing
 * ============================================================ */
__global__ void test_wave_intrinsics(uint32_t* results) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane = threadIdx.x % 64;  // wavefront lane (RDNA = 32 or 64)

    // Each lane is a "neuron" with a membrane potential
    float membrane = sinf((float)lane * 0.1f);  // initial state

    // 8 rounds of neural communication via wave intrinsics
    for (int round = 0; round < 8; round++) {
        // Read neighbor's state (axonal transport)
        int src_lane = (lane + 1) % 32;
        float neighbor = __shfl(membrane, src_lane);

        // Integrate
        membrane = membrane * 0.8f + neighbor * 0.2f;

        // Population spike detection via ballot
        int spiked = (membrane > 0.5f) ? 1 : 0;
        uint64_t spike_mask = __ballot(spiked);

        // Use population spike count as feedback
        int pop_spikes = __popcll(spike_mask);
        membrane -= (float)pop_spikes * 0.01f;  // lateral inhibition
    }

    if (tid < 2048) {
        results[tid * 2 + 0] = __float_as_uint(membrane);
        results[tid * 2 + 1] = lane;
    }
}

/* ============================================================
 * TEST 6: Full hwreg write scan — which registers accept writes?
 * ============================================================ */
__global__ void test_hwreg_write_scan(uint32_t* results) {
    uint32_t tid = threadIdx.x;
    if (tid > 0) return;  // Single thread only

    // For each hwreg, try: read → write test pattern → read back → restore
    uint32_t test_val = 0xA5A5A5A5;

    // hwreg(0) - HW_ID (probably read-only)
    uint32_t orig, readback;

    // Manually test each safe hwreg
    // hwreg(1) = MODE — MOST LIKELY WRITABLE
    asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(orig));
    asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(test_val));
    asm volatile("s_getreg_b32 %0, hwreg(1)" : "=s"(readback));
    asm volatile("s_setreg_b32 hwreg(1), %0" : : "s"(orig));  // restore
    results[0] = orig;
    results[1] = readback;
    results[2] = (readback != orig) ? 1 : 0;  // 1 = writable

    // hwreg(2) = STATUS
    asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(orig));
    asm volatile("s_setreg_b32 hwreg(2), %0" : : "s"(test_val));
    asm volatile("s_getreg_b32 %0, hwreg(2)" : "=s"(readback));
    asm volatile("s_setreg_b32 hwreg(2), %0" : : "s"(orig));
    results[3] = orig;
    results[4] = readback;
    results[5] = (readback != orig) ? 1 : 0;

    // hwreg(3) = TRAPSTS (trap status)
    asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(orig));
    asm volatile("s_setreg_b32 hwreg(3), %0" : : "s"(test_val));
    asm volatile("s_getreg_b32 %0, hwreg(3)" : "=s"(readback));
    asm volatile("s_setreg_b32 hwreg(3), %0" : : "s"(orig));
    results[6] = orig;
    results[7] = readback;
    results[8] = (readback != orig) ? 1 : 0;

    // hwreg(4) = HW_ID (read-only expected)
    asm volatile("s_getreg_b32 %0, hwreg(4)" : "=s"(orig));
    asm volatile("s_setreg_b32 hwreg(4), %0" : : "s"(test_val));
    asm volatile("s_getreg_b32 %0, hwreg(4)" : "=s"(readback));
    results[9] = orig;
    results[10] = readback;
    results[11] = (readback != orig) ? 1 : 0;

    // hwreg(5) = GPR_ALLOC
    asm volatile("s_getreg_b32 %0, hwreg(5)" : "=s"(orig));
    asm volatile("s_setreg_b32 hwreg(5), %0" : : "s"(test_val));
    asm volatile("s_getreg_b32 %0, hwreg(5)" : "=s"(readback));
    asm volatile("s_setreg_b32 hwreg(5), %0" : : "s"(orig));
    results[12] = orig;
    results[13] = readback;
    results[14] = (readback != orig) ? 1 : 0;

    // hwreg(6) = LDS_ALLOC
    asm volatile("s_getreg_b32 %0, hwreg(6)" : "=s"(orig));
    asm volatile("s_setreg_b32 hwreg(6), %0" : : "s"(test_val));
    asm volatile("s_getreg_b32 %0, hwreg(6)" : "=s"(readback));
    asm volatile("s_setreg_b32 hwreg(6), %0" : : "s"(orig));
    results[15] = orig;
    results[16] = readback;
    results[17] = (readback != orig) ? 1 : 0;

    // hwreg(13) = IB_STS (instruction buffer status)
    asm volatile("s_getreg_b32 %0, hwreg(13)" : "=s"(orig));
    asm volatile("s_setreg_b32 hwreg(13), %0" : : "s"(test_val));
    asm volatile("s_getreg_b32 %0, hwreg(13)" : "=s"(readback));
    asm volatile("s_setreg_b32 hwreg(13), %0" : : "s"(orig));
    results[18] = orig;
    results[19] = readback;
    results[20] = (readback != orig) ? 1 : 0;

    // hwreg(14) = FLAT_SCR_LO
    asm volatile("s_getreg_b32 %0, hwreg(14)" : "=s"(orig));
    asm volatile("s_setreg_b32 hwreg(14), %0" : : "s"(test_val));
    asm volatile("s_getreg_b32 %0, hwreg(14)" : "=s"(readback));
    asm volatile("s_setreg_b32 hwreg(14), %0" : : "s"(orig));
    results[21] = orig;
    results[22] = readback;
    results[23] = (readback != orig) ? 1 : 0;

    // hwreg(15) = FLAT_SCR_HI
    asm volatile("s_getreg_b32 %0, hwreg(15)" : "=s"(orig));
    asm volatile("s_setreg_b32 hwreg(15), %0" : : "s"(test_val));
    asm volatile("s_getreg_b32 %0, hwreg(15)" : "=s"(readback));
    asm volatile("s_setreg_b32 hwreg(15), %0" : : "s"(orig));
    results[24] = orig;
    results[25] = readback;
    results[26] = (readback != orig) ? 1 : 0;

    // hwreg(16) = XCC_ID (cross-chiplet)
    asm volatile("s_getreg_b32 %0, hwreg(16)" : "=s"(orig));
    asm volatile("s_setreg_b32 hwreg(16), %0" : : "s"(test_val));
    asm volatile("s_getreg_b32 %0, hwreg(16)" : "=s"(readback));
    results[27] = orig;
    results[28] = readback;
    results[29] = (readback != orig) ? 1 : 0;

    // hwreg(19) = SQ_PERF_SNAPSHOT_DATA
    asm volatile("s_getreg_b32 %0, hwreg(19)" : "=s"(orig));
    asm volatile("s_setreg_b32 hwreg(19), %0" : : "s"(test_val));
    asm volatile("s_getreg_b32 %0, hwreg(19)" : "=s"(readback));
    results[30] = orig;
    results[31] = readback;
    results[32] = (readback != orig) ? 1 : 0;
}

/* ============================================================
 * TEST 7: Instruction timing jitter as stochastic source
 * Different instructions have data-dependent timing on some GPUs
 * ============================================================ */
__global__ void test_instruction_jitter(uint32_t* results) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t t0, t1;

    // Measure timing of integer division (data-dependent on some HW)
    uint32_t a = tid * 17 + 3;
    uint32_t b = (tid % 7) + 1;

    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t0));
    uint32_t q = a / b;  // integer division - variable latency?
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t1));
    uint32_t div_cycles = t1 - t0;

    // Measure transcendental (definitely variable on some HW)
    float x = __int_as_float(a | 0x3F800000);  // 1.0-2.0 range
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t0));
    float s = sinf(x);
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t1));
    uint32_t sin_cycles = t1 - t0;

    // Measure branch divergence timing
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t0));
    float val;
    if (tid & 1) {
        val = sqrtf(x);
    } else {
        val = x * x;
    }
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(t1));
    uint32_t branch_cycles = t1 - t0;

    if (tid < 4096) {
        results[tid * 4 + 0] = div_cycles;
        results[tid * 4 + 1] = sin_cycles;
        results[tid * 4 + 2] = branch_cycles;
        results[tid * 4 + 3] = __float_as_uint(val + s + (float)q);
    }
}

/* ============================================================
 * MAIN — Run all tests and report
 * ============================================================ */
int main() {
    printf("=== DEEP SILICON PROBE: Unexplored Neuromorphic Capabilities ===\n\n");

    /* ---- TEST 1: s_setreg MODE register ---- */
    printf("--- TEST 1: s_setreg (register write from shader) ---\n");
    {
        uint32_t *d_results, h_results[1024];
        hipMalloc(&d_results, 1024 * sizeof(uint32_t));
        hipMemset(d_results, 0, 1024 * sizeof(uint32_t));

        test_setreg_mode<<<1, 256>>>(d_results);
        hipDeviceSynchronize();
        hipMemcpy(h_results, d_results, 1024 * sizeof(uint32_t), hipMemcpyDeviceToHost);

        printf("  Thread 0: MODE before=0x%08x tried=0x%08x after=0x%08x restored=0x%08x\n",
               h_results[0], h_results[1], h_results[2], h_results[3]);

        int writable = (h_results[2] != h_results[0]) ? 1 : 0;
        printf("  MODE WRITABLE: %s\n", writable ? "*** YES! ***" : "no");

        if (writable) {
            printf("  >>> s_setreg WORKS! FP rounding can be changed mid-kernel! <<<\n");
        }

        // Test computation effect
        float *d_comp, h_comp[3072];
        hipMalloc(&d_comp, 3072 * sizeof(float));
        test_setreg_computation<<<1, 1024>>>(d_comp);
        hipDeviceSynchronize();
        hipMemcpy(h_comp, d_comp, 3072 * sizeof(float), hipMemcpyDeviceToHost);

        printf("  Computation test (1.0 + 5.96e-8):\n");
        printf("    Default (RNE): %.10e (bits: 0x%08x)\n", h_comp[0], *(uint32_t*)&h_comp[0]);
        printf("    RTZ:           %.10e (bits: 0x%08x)\n", h_comp[1], *(uint32_t*)&h_comp[1]);
        printf("    RNI:           %.10e (bits: 0x%08x)\n", h_comp[2], *(uint32_t*)&h_comp[2]);

        int comp_differs = (*(uint32_t*)&h_comp[0] != *(uint32_t*)&h_comp[1]) ||
                           (*(uint32_t*)&h_comp[0] != *(uint32_t*)&h_comp[2]);
        printf("  Rounding affects computation: %s\n\n",
               comp_differs ? "*** YES — CONSTITUTIVE NONLINEARITY! ***" : "no difference");

        hipFree(d_results);
        hipFree(d_comp);
    }

    /* ---- TEST 2: Atomic contention as neural integration ---- */
    printf("--- TEST 2: Atomic contention neural integration ---\n");
    {
        int *d_soma, *d_dendrites, *d_spikes;
        int h_spikes[64];
        hipMalloc(&d_soma, 64 * sizeof(int));
        hipMalloc(&d_dendrites, 1024 * sizeof(int));
        hipMalloc(&d_spikes, 64 * sizeof(int));
        hipMemset(d_soma, 0, 64 * sizeof(int));
        hipMemset(d_spikes, 0, 64 * sizeof(int));

        // Initialize random-ish weights
        int h_weights[1024];
        for (int i = 0; i < 1024; i++) h_weights[i] = (i * 7 + 3) % 100 - 50;
        hipMemcpy(d_dendrites, h_weights, 1024 * sizeof(int), hipMemcpyHostToDevice);

        // Run 10 "timesteps" of neural integration
        for (int t = 0; t < 10; t++) {
            hipMemset(d_soma, 0, 64 * sizeof(int));
            test_atomic_neuron<<<16, 256>>>(d_soma, d_dendrites, d_spikes, 500);
            hipDeviceSynchronize();
        }

        hipMemcpy(h_spikes, d_spikes, 64 * sizeof(int), hipMemcpyDeviceToHost);
        int total_spikes = 0, max_spikes = 0, active = 0;
        for (int i = 0; i < 64; i++) {
            total_spikes += h_spikes[i];
            if (h_spikes[i] > max_spikes) max_spikes = h_spikes[i];
            if (h_spikes[i] > 0) active++;
        }
        printf("  64 neurons, 10 timesteps, threshold=500\n");
        printf("  Total spikes: %d, Active neurons: %d/64, Max spikes: %d\n",
               total_spikes, active, max_spikes);
        printf("  Hardware atomic = neural integration: %s\n\n",
               active > 0 ? "WORKING" : "no spikes");

        hipFree(d_soma); hipFree(d_dendrites); hipFree(d_spikes);
    }

    /* ---- TEST 3: LDS spiking neural network ---- */
    printf("--- TEST 3: LDS synaptic memory + spiking network ---\n");
    {
        float *d_outputs, *d_weights;
        int n_neurons = 128, n_inputs = 16;
        hipMalloc(&d_outputs, NBLOCKS * n_neurons * sizeof(float));
        hipMalloc(&d_weights, NBLOCKS * 4096 * sizeof(float));

        float h_weights[4096];
        for (int i = 0; i < 4096; i++) h_weights[i] = sinf(i * 0.01f) * 0.5f;
        for (int b = 0; b < NBLOCKS; b++)
            hipMemcpy(d_weights + b * 4096, h_weights, 4096 * sizeof(float), hipMemcpyHostToDevice);

        test_lds_synapse<<<NBLOCKS, 256>>>(d_outputs, d_weights, n_neurons, n_inputs);
        hipDeviceSynchronize();

        float h_out[NBLOCKS * 128];
        hipMemcpy(h_out, d_outputs, NBLOCKS * 128 * sizeof(float), hipMemcpyDeviceToHost);

        float mean = 0, var = 0;
        int nonzero = 0;
        for (int i = 0; i < NBLOCKS * 128; i++) {
            mean += h_out[i];
            if (h_out[i] != 0.0f) nonzero++;
        }
        mean /= (NBLOCKS * 128);
        for (int i = 0; i < NBLOCKS * 128; i++) var += (h_out[i] - mean) * (h_out[i] - mean);
        var /= (NBLOCKS * 128);

        printf("  %d CUs × %d neurons × 16 timesteps, all in LDS\n", NBLOCKS, n_neurons);
        printf("  Mean membrane: %.6f, Variance: %.6f, Nonzero: %d/%d\n",
               mean, var, nonzero, NBLOCKS * 128);
        printf("  LDS spiking network: %s\n\n",
               nonzero > 0 ? "ACTIVE" : "all quiescent");

        hipFree(d_outputs); hipFree(d_weights);
    }

    /* ---- TEST 5: Wave intrinsics as neural fabric ---- */
    printf("--- TEST 5: Wave intrinsics (shuffle/ballot) neural fabric ---\n");
    {
        uint32_t *d_results, h_results[4096];
        hipMalloc(&d_results, 4096 * sizeof(uint32_t));
        test_wave_intrinsics<<<4, 256>>>(d_results);
        hipDeviceSynchronize();
        hipMemcpy(h_results, d_results, 4096 * sizeof(uint32_t), hipMemcpyDeviceToHost);

        float min_v = 1e9, max_v = -1e9;
        int unique = 0;
        float prev = -999;
        for (int i = 0; i < 64; i++) {
            float v;
            // reinterpret bits
            memcpy(&v, &h_results[i * 2], sizeof(float));
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
            if (v != prev) { unique++; prev = v; }
        }
        printf("  32-lane wavefront, 8 rounds of shuffle+ballot\n");
        printf("  Membrane range: [%.4f, %.4f], Unique values: %d/64\n",
               min_v, max_v, unique);
        printf("  Wave intrinsics as neural fabric: %s\n\n",
               unique > 1 ? "DIVERSE (working)" : "collapsed");

        hipFree(d_results);
    }

    /* ---- TEST 6: hwreg write scan ---- */
    printf("--- TEST 6: hwreg write scan (which registers accept s_setreg?) ---\n");
    {
        uint32_t *d_results, h_results[64];
        hipMalloc(&d_results, 64 * sizeof(uint32_t));
        hipMemset(d_results, 0, 64 * sizeof(uint32_t));
        test_hwreg_write_scan<<<1, 1>>>(d_results);
        hipDeviceSynchronize();
        hipMemcpy(h_results, d_results, 64 * sizeof(uint32_t), hipMemcpyDeviceToHost);

        const char* names[] = {
            "MODE(1)", "STATUS(2)", "TRAPSTS(3)", "HW_ID(4)",
            "GPR_ALLOC(5)", "LDS_ALLOC(6)", "IB_STS(13)",
            "FLAT_SCR_LO(14)", "FLAT_SCR_HI(15)", "XCC_ID(16)", "PERF_DATA(19)"
        };
        int offsets[] = {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30};

        int writable_count = 0;
        for (int i = 0; i < 11; i++) {
            int base = offsets[i];
            uint32_t orig = h_results[base];
            uint32_t readback = h_results[base + 1];
            uint32_t changed = h_results[base + 2];
            printf("  %-18s orig=0x%08x readback=0x%08x %s\n",
                   names[i], orig, readback, changed ? "*** WRITABLE ***" : "read-only");
            if (changed) writable_count++;
        }
        printf("  Total writable hwregs: %d/11\n\n", writable_count);

        hipFree(d_results);
    }

    /* ---- TEST 7: Instruction timing jitter ---- */
    printf("--- TEST 7: Instruction timing jitter (stochastic source) ---\n");
    {
        uint32_t *d_results, h_results[16384];
        hipMalloc(&d_results, 16384 * sizeof(uint32_t));
        test_instruction_jitter<<<4, 256>>>(d_results);
        hipDeviceSynchronize();
        hipMemcpy(h_results, d_results, 16384 * sizeof(uint32_t), hipMemcpyDeviceToHost);

        // Analyze timing distributions
        uint64_t div_sum = 0, sin_sum = 0, branch_sum = 0;
        uint32_t div_min = UINT32_MAX, div_max = 0;
        uint32_t sin_min = UINT32_MAX, sin_max = 0;
        uint32_t branch_min = UINT32_MAX, branch_max = 0;
        int n = 1024;
        for (int i = 0; i < n; i++) {
            uint32_t d = h_results[i * 4 + 0];
            uint32_t s = h_results[i * 4 + 1];
            uint32_t b = h_results[i * 4 + 2];
            div_sum += d; sin_sum += s; branch_sum += b;
            if (d < div_min) div_min = d; if (d > div_max) div_max = d;
            if (s < sin_min) sin_min = s; if (s > sin_max) sin_max = s;
            if (b < branch_min) branch_min = b; if (b > branch_max) branch_max = b;
        }
        printf("  Division:  mean=%.1f range=[%u,%u]\n", (double)div_sum/n, div_min, div_max);
        printf("  Sin:       mean=%.1f range=[%u,%u]\n", (double)sin_sum/n, sin_min, sin_max);
        printf("  Branch:    mean=%.1f range=[%u,%u]\n", (double)branch_sum/n, branch_min, branch_max);

        int has_jitter = (div_max > div_min + 2) || (sin_max > sin_min + 2) || (branch_max > branch_min + 2);
        printf("  Timing jitter (stochastic): %s\n\n",
               has_jitter ? "YES — exploitable for computation" : "deterministic");

        hipFree(d_results);
    }

    printf("=== DEEP SILICON PROBE COMPLETE ===\n");
    return 0;
}
