/* GMP mirror of src/bin/bench_bigint.rs, for apples-to-apples comparison of
 * this crate's pure-Rust bigint kernels against GMP's assembly-backed ones.
 *
 * Build and run via scripts/bench_gmp.sh, which locates GMP on macOS
 * (Homebrew) and Linux.  Keep the operations and operand shapes in lockstep
 * with bench_bigint when either changes:
 *
 *   - mul_ref                       n x n multiply
 *   - mod_mul (odd modulus)         n x n multiply, one reduction
 *   - montgomery_pow (e=65537)      modexp, F4 public exponent
 *   - montgomery_pow (random 256b)  modexp, random 256-bit exponent
 *   - div_rem / modulo              n divided by n/2
 *
 * Operands are drawn once per size with the top bit forced, exactly as the
 * Rust harness does; the DRBG streams differ, so compare distributions, not
 * individual draws.  Usage mirrors bench_bigint: sizes in bits as arguments,
 * with the same defaults when none are given.
 */
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* Same pacing as bench_bigint: double the iteration count until the timed
 * region reaches the target, then report the mean. */
#define TARGET_NS 2e8
#define MAX_ITERS 65536L

#define BENCH(label, min_iters, op) do {                             \
    long iters = (min_iters);                                        \
    double ns;                                                       \
    for (;;) {                                                       \
        double t0 = now_ns();                                        \
        for (long k = 0; k < iters; k++) { op; }                     \
        ns = now_ns() - t0;                                          \
        if (ns >= TARGET_NS || iters >= MAX_ITERS) break;            \
        iters *= 2;                                                  \
        if (iters > MAX_ITERS) iters = MAX_ITERS;                    \
    }                                                                \
    printf("| %s | %.1f | %ld |\n", label, ns / (double)iters, iters); \
} while (0)

int main(int argc, char **argv) {
    static const int default_bits[] = {256, 512, 1024, 2048, 4096};
    int sizes[64];
    int n_sizes = 0;

    for (int i = 1; i < argc && n_sizes < 64; i++) {
        int bits = atoi(argv[i]);
        if (bits > 0) sizes[n_sizes++] = bits;
    }
    if (n_sizes == 0) {
        for (int i = 0; i < 5; i++) sizes[n_sizes++] = default_bits[i];
    }

    gmp_randstate_t rs;
    gmp_randinit_default(rs);
    gmp_randseed_ui(rs, 0x4d4d4d4dUL);

    mpz_t lhs, rhs, mod, base, divisor, e65537, erand, out, out2;
    mpz_inits(lhs, rhs, mod, base, divisor, e65537, erand, out, out2, NULL);
    mpz_set_ui(e65537, 65537);

    printf("# GMP %s bigint microbenchmarks\n", gmp_version);
    printf("Columns: nanoseconds per operation and iterations used.\n");

    for (int s = 0; s < n_sizes; s++) {
        int bits = sizes[s];
        mpz_urandomb(lhs, rs, bits);
        mpz_setbit(lhs, bits - 1);
        mpz_urandomb(rhs, rs, bits);
        mpz_setbit(rhs, bits - 1);
        mpz_urandomb(mod, rs, bits);
        mpz_setbit(mod, bits - 1);
        mpz_setbit(mod, 0); /* odd, as the Montgomery paths require */
        mpz_urandomb(base, rs, bits);
        mpz_setbit(base, bits - 1);
        mpz_urandomb(divisor, rs, bits / 2);
        mpz_setbit(divisor, bits / 2 - 1);
        mpz_urandomb(erand, rs, 256);
        mpz_setbit(erand, 255);

        printf("\n### %d-bit\n", bits);
        printf("| Operation | ns/op | Iters |\n");
        printf("|-----------|------:|------:|\n");
        BENCH("mul_ref", 2, mpz_mul(out, lhs, rhs));
        BENCH("mod_mul (odd modulus)", 2,
              { mpz_mul(out, lhs, rhs); mpz_mod(out, out, mod); });
        BENCH("montgomery_pow (e=65537)", 1, mpz_powm(out, base, e65537, mod));
        BENCH("montgomery_pow (random 256b e)", 1,
              mpz_powm(out, base, erand, mod));
        BENCH("div_rem", 1, mpz_tdiv_qr(out, out2, lhs, divisor));
        BENCH("modulo", 1, mpz_mod(out, lhs, divisor));
    }

    mpz_clears(lhs, rhs, mod, base, divisor, e65537, erand, out, out2, NULL);
    gmp_randclear(rs);
    return 0;
}
