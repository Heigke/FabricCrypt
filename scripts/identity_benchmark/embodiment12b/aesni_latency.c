/* Task C — AES-NI single-instruction latency via TSC bracketing.
 * Usage: ./aesni_latency <n> > out.bin (n int32 cycle counts)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <wmmintrin.h>
#include <emmintrin.h>

static inline uint64_t rdtscp(void) {
    unsigned aux, lo, hi;
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux) :: "memory");
    return ((uint64_t)hi << 32) | lo;
}

int main(int argc, char**argv) {
    int n = (argc>1)? atoi(argv[1]) : 100000;
    uint32_t *out = malloc(sizeof(uint32_t)*n);
    __m128i state = _mm_set_epi32(0x12345678, 0x9abcdef0, 0x0f0f0f0f, 0xf0f0f0f0);
    __m128i key   = _mm_set_epi32(0xdeadbeef, 0xcafebabe, 0x01234567, 0x89abcdef);

    /* warmup */
    for (int i=0;i<10000;i++) state = _mm_aesenc_si128(state, key);

    for (int i=0;i<n;i++) {
        __asm__ __volatile__("" ::: "memory");
        uint64_t a = rdtscp();
        state = _mm_aesenc_si128(state, key);
        __asm__ __volatile__("" : "+x"(state) :: "memory");
        uint64_t b = rdtscp();
        uint64_t d = b - a;
        out[i] = (d > 0xFFFFFFFFULL) ? 0xFFFFFFFFu : (uint32_t)d;
    }
    /* keep state live */
    if (((uint32_t*)&state)[0] == 0xdeadbeef) fprintf(stderr,"?\n");
    fwrite(out, sizeof(uint32_t), n, stdout);
    free(out);
    return 0;
}
