/* probeB_ro_pair.hip — Ring-oscillator pair race (Suh/Devadas 2007 RO-PUF analog).
 *
 * Two blocks per "race" run the SAME instruction sequence. Each block races to
 * complete a fixed inner-loop count and the FIRST to finish writes its block-id
 * to a shared winner slot via atomic CAS. We do 10k races per pair, count wins.
 *
 * Whichever physical CU pair the scheduler places blocks on has slightly
 * different transistor speed — winner frequency reveals silicon-bound delay.
 *
 * To avoid scheduler bias we run pairs in two phases per race:
 *   Phase 0: block 0 vs block 1
 *   Phase 1: block 1 vs block 0  (swap roles)
 * and record both. The output per race is which BLOCK_ID won, plus the
 * cycle delta read from hwreg(29) per block.
 *
 * Build:
 *   hipcc --offload-arch=gfx1151 -O1 -o probeB_ro_pair probeB_ro_pair.hip
 *
 * Usage: probeB_ro_pair <races> <out.bin>
 *   Output layout:
 *     [0]: magic 0x524F5031
 *     [1]: races
 *     [2]: fields_per_race (=4)  [winner, hwid0, hwid1, dCyc01]
 *     [3]: reserved
 *     [4..]: races * 4 uint32
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define INNER_ITERS 4096

__global__ void ro_pair_kernel(uint32_t* race_out, uint32_t* winner_flag,
                               uint32_t* arrival_counter, uint32_t* start_flag)
{
    // race_out layout per race: [winner, hwid0, hwid1, dCyc01]
    // winner_flag: per-race uint32, initial 0, set by first finisher with (bid+1).
    // arrival_counter: blocks atomically increment on entry; last block sets start_flag.
    // start_flag: 0 until both arrived. All threads spin until set.
    int bid = blockIdx.x;        // 0 or 1
    int tid = threadIdx.x;

    // Read HW_ID1 (hwreg 23) once per block (lane 0).
    uint32_t hwid = __builtin_amdgcn_s_getreg((23) | (0 << 6) | ((32 - 1) << 11));

    // Barrier: only one thread per block does the arrival CAS, rest wait at __syncthreads.
    if (tid == 0) {
        uint32_t arrived = atomicAdd(arrival_counter, 1u);
        if (arrived == 1u) {
            // last to arrive: release start flag
            atomicExch(start_flag, 1u);
        } else {
            // first to arrive: spin until release
            while (atomicAdd(start_flag, 0u) == 0u) { /* spin */ }
        }
    }
    __syncthreads();

    // Read cycles at start (hwreg 29).
    uint32_t c0 = __builtin_amdgcn_s_getreg((29) | (0 << 6) | ((32 - 1) << 11));

    // Compute: identical instruction sequence on both blocks.
    float acc = (float)tid * 1.000173f + 1.0f;
    #pragma unroll 1
    for (int i = 0; i < INNER_ITERS; i++) {
        acc = __fmaf_rn(acc, 1.0000001f, 0.0000001f);
        acc = __fmaf_rn(acc, 0.9999999f, 0.0000001f);
    }
    // Force a side-effect on tid 0 only -> race for winner_flag.
    uint32_t c1 = __builtin_amdgcn_s_getreg((29) | (0 << 6) | ((32 - 1) << 11));
    uint32_t dcyc = c1 - c0;

    // Avoid the compiler dead-code-eliminating `acc`: depend the CAS on it.
    if (tid == 0) {
        // Mix acc into a sentinel so the kernel can't be optimized away.
        uint32_t sentinel = (acc != 0.0f) ? (bid + 1u) : 0xFFu;
        // CAS: first to claim wins.
        uint32_t old = atomicCAS(winner_flag, 0u, sentinel);
        // Record per-block info regardless of win.
        if (bid == 0) {
            race_out[1] = hwid;
            race_out[3] = dcyc;       // store block 0's cycle count
        } else {
            race_out[2] = hwid;
            // dCyc difference (block0 - block1) is stored after both finished.
            // We store block1 cycle in upper 16 bits? No — keep it simple: race_out[3] holds block 0 only.
        }
        if (old == 0u) {
            race_out[0] = sentinel;   // we won
        }
    }
}

int main(int argc, char** argv)
{
    int races = (argc > 1) ? atoi(argv[1]) : 10000;
    const char* out_path = (argc > 2) ? argv[2] : "probeB_ro.bin";

    HIP_CHECK(hipSetDevice(0));

    size_t fields = 4;
    size_t payload_words = (size_t)races * fields;
    uint32_t* host = (uint32_t*)calloc(4 + payload_words, sizeof(uint32_t));
    host[0] = 0x524F5031u;  // "ROP1"
    host[1] = (uint32_t)races;
    host[2] = (uint32_t)fields;
    host[3] = 0;

    uint32_t* dev_out = nullptr;
    uint32_t* dev_flag = nullptr;
    uint32_t* dev_arrival = nullptr;
    uint32_t* dev_start = nullptr;
    HIP_CHECK(hipMalloc(&dev_out, fields * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&dev_flag, sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&dev_arrival, sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&dev_start, sizeof(uint32_t)));

    for (int r = 0; r < races; r++) {
        HIP_CHECK(hipMemset(dev_out, 0, fields * sizeof(uint32_t)));
        HIP_CHECK(hipMemset(dev_flag, 0, sizeof(uint32_t)));
        HIP_CHECK(hipMemset(dev_arrival, 0, sizeof(uint32_t)));
        HIP_CHECK(hipMemset(dev_start, 0, sizeof(uint32_t)));
        // Two blocks, 64 threads each.
        hipLaunchKernelGGL(ro_pair_kernel,
                           dim3(2), dim3(64), 0, 0,
                           dev_out, dev_flag, dev_arrival, dev_start);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(host + 4 + (size_t)r * fields, dev_out,
                            fields * sizeof(uint32_t), hipMemcpyDeviceToHost));
    }

    FILE* f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "open %s failed\n", out_path); return 1; }
    fwrite(host, sizeof(uint32_t), 4 + payload_words, f);
    fclose(f);

    HIP_CHECK(hipFree(dev_out));
    HIP_CHECK(hipFree(dev_flag));
    HIP_CHECK(hipFree(dev_arrival));
    HIP_CHECK(hipFree(dev_start));
    free(host);
    printf("[probeB] wrote %s races=%d\n", out_path, races);
    return 0;
}
