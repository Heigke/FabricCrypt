/* C_tournament_ro.hip — Tournament aggregator over RO-pair races.
 *
 * Runs `slots` slots; each slot runs `races` 2-block races (same kernel as
 * probeB_ro_pair). Per slot the winner-count for block-id 1 (vs 2) is the
 * aggregated tournament bit (>races/2 → bit=1). Output is per-slot win counts.
 *
 * Build: hipcc --offload-arch=gfx1151 -O1 -o C_tournament_ro C_tournament_ro.hip
 * Usage: C_tournament_ro <slots> <races_per_slot> <out.bin>
 *   Output: magic 0x544F5552 ("TOUR"), slots, races_per_slot, reserved,
 *           then slots * uint32 (count of block-1 wins per slot)
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <thread>

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define INNER_ITERS 4096

__global__ void ro_pair_kernel(uint32_t* winner_flag,
                               uint32_t* arrival_counter,
                               uint32_t* start_flag)
{
    int bid = blockIdx.x;        // 0 or 1
    int tid = threadIdx.x;
    if (tid == 0) {
        uint32_t arrived = atomicAdd(arrival_counter, 1u);
        if (arrived == 1u) {
            atomicExch(start_flag, 1u);
        } else {
            while (atomicAdd(start_flag, 0u) == 0u) { /* spin */ }
        }
    }
    __syncthreads();
    float acc = (float)tid * 1.000173f + 1.0f;
    #pragma unroll 1
    for (int i = 0; i < INNER_ITERS; i++) {
        acc = __fmaf_rn(acc, 1.0000001f, 0.0000001f);
        acc = __fmaf_rn(acc, 0.9999999f, 0.0000001f);
    }
    if (tid == 0) {
        uint32_t sentinel = (acc != 0.0f) ? (bid + 1u) : 0xFFu;
        atomicCAS(winner_flag, 0u, sentinel);
    }
}

int main(int argc, char** argv)
{
    int slots = (argc > 1) ? atoi(argv[1]) : 79;
    int races = (argc > 2) ? atoi(argv[2]) : 256;
    const char* out_path = (argc > 3) ? argv[3] : "C_tournament.bin";

    HIP_CHECK(hipSetDevice(0));

    uint32_t* dev_flag = nullptr;
    uint32_t* dev_arrival = nullptr;
    uint32_t* dev_start = nullptr;
    HIP_CHECK(hipMalloc(&dev_flag, sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&dev_arrival, sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&dev_start, sizeof(uint32_t)));

    uint32_t* host_counts = (uint32_t*)calloc(4 + slots, sizeof(uint32_t));
    host_counts[0] = 0x544F5552u;
    host_counts[1] = (uint32_t)slots;
    host_counts[2] = (uint32_t)races;
    host_counts[3] = 0;

    uint32_t winner = 0;
    for (int s = 0; s < slots; s++) {
        uint32_t b1_wins = 0;  // count wins of block-id 1 (sentinel == 2)
        for (int r = 0; r < races; r++) {
            HIP_CHECK(hipMemset(dev_flag, 0, sizeof(uint32_t)));
            HIP_CHECK(hipMemset(dev_arrival, 0, sizeof(uint32_t)));
            HIP_CHECK(hipMemset(dev_start, 0, sizeof(uint32_t)));
            hipLaunchKernelGGL(ro_pair_kernel, dim3(2), dim3(64), 0, 0,
                               dev_flag, dev_arrival, dev_start);
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(&winner, dev_flag, sizeof(uint32_t), hipMemcpyDeviceToHost));
            if (winner == 2u) b1_wins++;
        }
        host_counts[4 + s] = b1_wins;
        fprintf(stderr, "[slot %d/%d] b1_wins=%u/%d\n", s+1, slots, b1_wins, races);
    }

    FILE* f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "open %s failed\n", out_path); return 1; }
    fwrite(host_counts, sizeof(uint32_t), 4 + slots, f);
    fclose(f);

    HIP_CHECK(hipFree(dev_flag));
    HIP_CHECK(hipFree(dev_arrival));
    HIP_CHECK(hipFree(dev_start));
    free(host_counts);
    printf("[C_tournament] wrote %s slots=%d races=%d\n", out_path, slots, races);
    return 0;
}
