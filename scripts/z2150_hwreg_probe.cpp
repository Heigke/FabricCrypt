// z2150_hwreg_probe.cpp — Read analog-adjacent hardware registers from GPU wavefront
// Target: AMD gfx1151 (RDNA 3.5, Radeon 8060S) via HIP + inline s_getreg_b32
// Requires: HSA_OVERRIDE_GFX_VERSION=11.0.0
//
// hwreg IDs (GFX11, from LLVM SIDefines.h):
//   1=MODE, 2=STATUS, 3=TRAPSTS, 5=GPR_ALLOC, 6=LDS_ALLOC,
//   7=IB_STS, 15=MEM_BASES, 23=HW_ID1, 24=HW_ID2,
//   25=POPS_PACKER, 27=PERF_SNAPSHOT_DATA, 28=IB_STS2
//
// s_getreg_b32 encoding: hwreg(id, offset, width)
//   default offset=0, width=32 → hwreg(id, 0, 31) in ISA encoding
//   In inline asm: hwreg(ID) reads full 32 bits

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(call)                                                        \
    do {                                                                        \
        hipError_t err = (call);                                                \
        if (err != hipSuccess) {                                                \
            fprintf(stderr, "HIP error at %s:%d — %s (%d)\n",                 \
                    __FILE__, __LINE__, hipGetErrorString(err), err);           \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// Each wavefront writes one row of register values
// We read 12 registers + 2 timestamps = 14 values per wavefront
#define NREGS 14

// Struct for one wavefront's register snapshot
struct WavefrontRegs {
    uint32_t mode;              // hwreg(1)  — rounding mode, exceptions, etc.
    uint32_t status;            // hwreg(2)  — wave status bits
    uint32_t trapsts;           // hwreg(3)  — trap/exception status
    uint32_t gpr_alloc;         // hwreg(5)  — VGPR/SGPR allocation for this wave
    uint32_t lds_alloc;         // hwreg(6)  — LDS allocation
    uint32_t ib_sts;            // hwreg(7)  — indirect buffer status
    uint32_t mem_bases;         // hwreg(15) — memory base pointers
    uint32_t hw_id1;            // hwreg(23) — CU, SE, SH, wave-in-SIMD
    uint32_t hw_id2;            // hwreg(24) — queue, pipe, ME, node
    uint32_t pops_packer;       // hwreg(25) — primitive ordered pixel shader packer
    uint32_t perf_snapshot;     // hwreg(27) — performance snapshot data (gfx11!)
    uint32_t ib_sts2;           // hwreg(28) — indirect buffer status 2
    uint32_t memtime_lo;        // s_memtime low 32 bits (GPU cycle counter)
    uint32_t memtime_hi;        // s_memtime high 32 bits
};

// GPU kernel: each wavefront reads its own hardware registers
__global__ void hwreg_probe_kernel(WavefrontRegs* out, int max_waves) {
    // Use block/thread IDs to compute a unique wavefront index
    // On RDNA 3.5, wavefront size = 32
    int wave_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;

    // Only lane 0 of each wavefront writes (all lanes read same SGPR value)
    if (lane != 0) return;
    if (wave_id >= max_waves) return;

    WavefrontRegs regs;

    // Read all hardware registers via s_getreg_b32
    // The "=s" constraint puts result in an SGPR (scalar register)
    uint32_t val;

    asm volatile("s_getreg_b32 %0, hwreg(1)"  : "=s"(val)); regs.mode = val;
    asm volatile("s_getreg_b32 %0, hwreg(2)"  : "=s"(val)); regs.status = val;
    asm volatile("s_getreg_b32 %0, hwreg(3)"  : "=s"(val)); regs.trapsts = val;
    asm volatile("s_getreg_b32 %0, hwreg(5)"  : "=s"(val)); regs.gpr_alloc = val;
    asm volatile("s_getreg_b32 %0, hwreg(6)"  : "=s"(val)); regs.lds_alloc = val;
    asm volatile("s_getreg_b32 %0, hwreg(7)"  : "=s"(val)); regs.ib_sts = val;
    asm volatile("s_getreg_b32 %0, hwreg(15)" : "=s"(val)); regs.mem_bases = val;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(val)); regs.hw_id1 = val;
    asm volatile("s_getreg_b32 %0, hwreg(24)" : "=s"(val)); regs.hw_id2 = val;
    asm volatile("s_getreg_b32 %0, hwreg(25)" : "=s"(val)); regs.pops_packer = val;
    asm volatile("s_getreg_b32 %0, hwreg(27)" : "=s"(val)); regs.perf_snapshot = val;
    asm volatile("s_getreg_b32 %0, hwreg(28)" : "=s"(val)); regs.ib_sts2 = val;

    // GFX11 removed s_memtime. Use s_sendmsg_rtn_b32 to read GPU realtime clock.
    // MSG_RTN_GET_REALTIME = 0x83 returns 64-bit realtime counter.
    // Alternatively, use __builtin_amdgcn_s_sendmsg_rtn or just read SHADER_CYCLES.
    // For simplicity, use __clock() intrinsic which maps to s_getreg SHADER_CYCLES on gfx11.
    // We also try reading the timestamp via s_getreg_b32 on the SHADER_CYCLES hwreg.
    // SHADER_CYCLES on GFX11 is accessible via __builtin_amdgcn_s_getreg with appropriate ID.
    //
    // Fallback: use __builtin_amdgcn_s_sendmsg_rtnl for realtime clock
    uint32_t cycles_lo, cycles_hi;
    // s_sendmsg_rtn_b32 with msg=0x83 (GET_REALTIME) — two calls for lo and hi
    // Actually, on gfx11 the simplest approach: use clock() or wall_clock
    asm volatile("s_getreg_b32 %0, hwreg(29)" : "=s"(cycles_lo)); // hwreg(29) = SHADER_CYCLES on some GFX11
    cycles_hi = 0; // SHADER_CYCLES is 32-bit only
    regs.memtime_lo = cycles_lo;
    regs.memtime_hi = cycles_hi;

    out[wave_id] = regs;
}

// Decode HW_ID1 fields (RDNA 3 / GFX11)
// Bits [3:0]   = WAVE_ID (wave slot within SIMD)
// Bits [5:4]   = SIMD_ID
// Bits [9:6]   = WGP_ID (workgroup processor)
// Bits [12:10] = SA_ID (shader array)
// Bits [15:13] = SE_ID (shader engine)
void decode_hw_id1(uint32_t hw_id1) {
    uint32_t wave_id = (hw_id1 >> 0)  & 0xF;
    uint32_t simd_id = (hw_id1 >> 4)  & 0x3;
    uint32_t wgp_id  = (hw_id1 >> 6)  & 0xF;
    uint32_t sa_id   = (hw_id1 >> 10) & 0x7;
    uint32_t se_id   = (hw_id1 >> 13) & 0x7;
    printf("    HW_ID1 decode: SE=%u SA=%u WGP=%u SIMD=%u WAVE=%u\n",
           se_id, sa_id, wgp_id, simd_id, wave_id);
}

// Decode HW_ID2 fields (RDNA 3 / GFX11)
// Bits [3:0]   = QUEUE_ID
// Bits [5:4]   = PIPE_ID
// Bits [7:6]   = ME_ID
// Bits [11:8]  = reserved
// Bits [15:12] = VM_ID (VMID for this wave)
void decode_hw_id2(uint32_t hw_id2) {
    uint32_t queue_id = (hw_id2 >> 0)  & 0xF;
    uint32_t pipe_id  = (hw_id2 >> 4)  & 0x3;
    uint32_t me_id    = (hw_id2 >> 6)  & 0x3;
    uint32_t vm_id    = (hw_id2 >> 12) & 0xF;
    printf("    HW_ID2 decode: QUEUE=%u PIPE=%u ME=%u VMID=%u\n",
           queue_id, pipe_id, me_id, vm_id);
}

// Decode GPR_ALLOC (GFX11)
// Bits [5:0]   = VGPR_BASE (in units of 4 VGPRs)
// Bits [13:8]  = VGPR_SIZE (in units of 4 VGPRs, 0 = 4 VGPRs)
// Bits [19:16] = SGPR_BASE (in units of 16 SGPRs)
// Bits [23:20] = SGPR_SIZE (in units of 16 SGPRs)
void decode_gpr_alloc(uint32_t gpr) {
    uint32_t vgpr_base = (gpr >> 0) & 0x3F;
    uint32_t vgpr_size = (gpr >> 8) & 0x3F;
    uint32_t sgpr_base = (gpr >> 16) & 0xF;
    uint32_t sgpr_size = (gpr >> 20) & 0xF;
    printf("    GPR_ALLOC: VGPR_BASE=%u (*4=%u) VGPR_SIZE=%u (*4=%u) "
           "SGPR_BASE=%u (*16=%u) SGPR_SIZE=%u (*16=%u)\n",
           vgpr_base, vgpr_base*4, vgpr_size, (vgpr_size+1)*4,
           sgpr_base, sgpr_base*16, sgpr_size, (sgpr_size+1)*16);
}

// Decode MODE register (GFX11)
void decode_mode(uint32_t mode) {
    uint32_t fp_round  = (mode >> 0) & 0xF;   // FP rounding mode
    uint32_t fp_denorm = (mode >> 4) & 0xF;    // FP denorm handling
    uint32_t dx10_clamp = (mode >> 8) & 0x1;
    uint32_t ieee      = (mode >> 9) & 0x1;
    uint32_t lod_clamped = (mode >> 10) & 0x1;
    printf("    MODE: FP_ROUND=0x%X FP_DENORM=0x%X DX10_CLAMP=%u IEEE=%u LOD_CLAMPED=%u\n",
           fp_round, fp_denorm, dx10_clamp, ieee, lod_clamped);
}

int main(int argc, char** argv) {
    printf("=== z2150: HIP hwreg Probe — gfx1151 RDNA 3.5 ===\n\n");

    // Check GPU
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    printf("GPU devices: %d\n", device_count);

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));
    printf("Device 0: %s (gcnArchName: %s)\n", props.name, props.gcnArchName);
    printf("CUs: %d, Max clock: %d MHz, Wavefront size: %d\n",
           props.multiProcessorCount, props.clockRate/1000, props.warpSize);
    printf("Memory: %.1f GB\n\n", props.totalGlobalMem / (1024.0*1024*1024));

    // Launch parameters
    int num_blocks = 4;       // 4 blocks
    int threads_per_block = 64; // 64 threads = 2 wavefronts of 32
    int total_threads = num_blocks * threads_per_block;
    int max_waves = total_threads / props.warpSize;
    printf("Launching %d blocks x %d threads = %d wavefronts\n\n",
           num_blocks, threads_per_block, max_waves);

    // Allocate device and host memory
    WavefrontRegs* d_regs = nullptr;
    WavefrontRegs* h_regs = nullptr;
    size_t buf_size = max_waves * sizeof(WavefrontRegs);

    HIP_CHECK(hipMalloc(&d_regs, buf_size));
    h_regs = (WavefrontRegs*)malloc(buf_size);
    memset(h_regs, 0, buf_size);
    HIP_CHECK(hipMemset(d_regs, 0, buf_size));

    // Launch kernel
    hipLaunchKernelGGL(hwreg_probe_kernel, dim3(num_blocks), dim3(threads_per_block),
                       0, 0, d_regs, max_waves);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Copy results back
    HIP_CHECK(hipMemcpy(h_regs, d_regs, buf_size, hipMemcpyDeviceToHost));

    // Print results
    printf("========== WAVEFRONT REGISTER SNAPSHOTS ==========\n\n");
    for (int w = 0; w < max_waves; w++) {
        WavefrontRegs& r = h_regs[w];
        uint64_t memtime = ((uint64_t)r.memtime_hi << 32) | r.memtime_lo;

        printf("--- Wavefront %d ---\n", w);
        printf("  MODE             = 0x%08X\n", r.mode);
        decode_mode(r.mode);
        printf("  STATUS           = 0x%08X\n", r.status);
        printf("  TRAPSTS          = 0x%08X\n", r.trapsts);
        printf("  GPR_ALLOC        = 0x%08X\n", r.gpr_alloc);
        decode_gpr_alloc(r.gpr_alloc);
        printf("  LDS_ALLOC        = 0x%08X\n", r.lds_alloc);
        printf("  IB_STS           = 0x%08X\n", r.ib_sts);
        printf("  MEM_BASES        = 0x%08X\n", r.mem_bases);
        printf("  HW_ID1           = 0x%08X\n", r.hw_id1);
        decode_hw_id1(r.hw_id1);
        printf("  HW_ID2           = 0x%08X\n", r.hw_id2);
        decode_hw_id2(r.hw_id2);
        printf("  POPS_PACKER      = 0x%08X\n", r.pops_packer);
        printf("  PERF_SNAPSHOT    = 0x%08X\n", r.perf_snapshot);
        printf("  IB_STS2          = 0x%08X\n", r.ib_sts2);
        printf("  s_memtime        = %lu (0x%016lX)\n", memtime, memtime);
        printf("\n");
    }

    // Summary: check for diversity (analog physics signal)
    printf("========== DIVERSITY ANALYSIS ==========\n");
    printf("(Different values across wavefronts indicate real hardware state)\n\n");

    // Check how many unique HW_ID1 values we see
    int unique_hw_id1 = 0;
    uint32_t seen_hw_id1[32] = {0};
    for (int w = 0; w < max_waves; w++) {
        bool found = false;
        for (int j = 0; j < unique_hw_id1; j++) {
            if (seen_hw_id1[j] == h_regs[w].hw_id1) { found = true; break; }
        }
        if (!found && unique_hw_id1 < 32) {
            seen_hw_id1[unique_hw_id1++] = h_regs[w].hw_id1;
        }
    }
    printf("Unique HW_ID1 values: %d / %d wavefronts\n", unique_hw_id1, max_waves);

    // Check memtime spread
    uint64_t min_t = UINT64_MAX, max_t = 0;
    for (int w = 0; w < max_waves; w++) {
        uint64_t t = ((uint64_t)h_regs[w].memtime_hi << 32) | h_regs[w].memtime_lo;
        if (t < min_t) min_t = t;
        if (t > max_t) max_t = t;
    }
    printf("s_memtime spread: %lu cycles (min=%lu, max=%lu)\n",
           max_t - min_t, min_t, max_t);

    // Check perf_snapshot diversity
    int unique_perf = 0;
    uint32_t seen_perf[32] = {0};
    for (int w = 0; w < max_waves; w++) {
        bool found = false;
        for (int j = 0; j < unique_perf; j++) {
            if (seen_perf[j] == h_regs[w].perf_snapshot) { found = true; break; }
        }
        if (!found && unique_perf < 32) {
            seen_perf[unique_perf++] = h_regs[w].perf_snapshot;
        }
    }
    printf("Unique PERF_SNAPSHOT values: %d / %d wavefronts\n", unique_perf, max_waves);

    printf("\n=== z2150 COMPLETE ===\n");

    // Cleanup
    free(h_regs);
    HIP_CHECK(hipFree(d_regs));
    return 0;
}
