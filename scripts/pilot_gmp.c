/* GMP counterpart of pilot_mp: the same operations, operand shapes, and
 * fresh-random-per-invocation protocol, so pilot-bench drives both identically
 * and PERFORMANCE.md can put rump and GMP side by side.
 *
 *   pilot_gmp <op>     draw ONE fresh random operand, run the GMP primitive
 *                      enough times to beat the clock, print ms/op
 *   pilot_gmp --list   every operation name
 *
 * Only the operations with a genuine GMP mpz counterpart are here. rump's
 * Montgomery domain (mul_mont/pow), mod_sqrt, and GF(2^m) have no mpz
 * equivalent and are intentionally absent.
 *
 * Build via scripts/bench_gmp.sh (which also builds bench_gmp).
 */
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── operands, drawn once from an OS-seeded state (fresh per process) ── */
typedef struct {
    mpz_t a, b, divisor, modulus, e_rand, q, r, g, s, t, tmp;
    int prime_reps;
} Pool;

static void pool_init(Pool *p, unsigned long bits) {
    gmp_randstate_t st;
    gmp_randinit_default(st);
    /* Fresh seed per process: OS clock XOR pid. */
    gmp_randseed_ui(st, (unsigned long)(now_s() * 1e9) ^ ((unsigned long)getpid() * 2654435761UL));

    mpz_inits(p->a, p->b, p->divisor, p->modulus, p->e_rand, p->q, p->r, p->g, p->s, p->t,
              p->tmp, NULL);
    mpz_urandomb(p->a, st, bits);
    mpz_setbit(p->a, bits - 1);
    mpz_urandomb(p->b, st, bits);
    mpz_setbit(p->b, bits - 1);
    mpz_urandomb(p->divisor, st, bits / 2); /* half-width, real quotient work */
    mpz_setbit(p->divisor, bits / 2 - 1);
    mpz_setbit(p->divisor, 0);
    mpz_urandomb(p->modulus, st, bits);
    mpz_setbit(p->modulus, bits - 1);
    mpz_setbit(p->modulus, 0); /* odd */
    mpz_urandomb(p->e_rand, st, 256);
    mpz_setbit(p->e_rand, 255);
    p->prime_reps = 15; /* probabilistic MR rounds for mpz_probab_prime_p */
    gmp_randclear(st);
}

/* Sink for scalar-returning ops (jacobi, isprime): without it, -O2 sees the
 * result unused and no memory written, and dead-code-eliminates the whole
 * call. `volatile` forces the store, keeping the call — the C analogue of
 * rump's black_box(). The value-writing ops (mpz_add/mul/…) are kept anyway
 * because they store into heap-allocated mpz_t limbs. */
static volatile long g_sink;

/* ── one operation on the pool ── */
static void run(const char *op, Pool *p) {
    if (!strcmp(op, "add"))          mpz_add(p->r, p->a, p->b);
    else if (!strcmp(op, "sub")) {   /* order so it never goes negative */
        if (mpz_cmp(p->a, p->b) >= 0) mpz_sub(p->r, p->a, p->b);
        else                          mpz_sub(p->r, p->b, p->a);
    }
    else if (!strcmp(op, "mul"))     mpz_mul(p->r, p->a, p->b);
    else if (!strcmp(op, "sqr"))     mpz_mul(p->r, p->a, p->a);
    else if (!strcmp(op, "divrem"))  mpz_tdiv_qr(p->q, p->r, p->a, p->divisor);
    else if (!strcmp(op, "modulo"))  mpz_mod(p->r, p->a, p->divisor);
    else if (!strcmp(op, "modmul")) { mpz_mul(p->tmp, p->a, p->b); mpz_mod(p->r, p->tmp, p->modulus); }
    else if (!strcmp(op, "modpow"))  mpz_powm(p->r, p->a, p->e_rand, p->modulus);
    else if (!strcmp(op, "gcd"))     mpz_gcd(p->r, p->a, p->b);
    else if (!strcmp(op, "gcdext"))  mpz_gcdext(p->g, p->s, p->t, p->a, p->b);
    else if (!strcmp(op, "modinv"))  mpz_invert(p->r, p->a, p->modulus);
    else if (!strcmp(op, "jacobi"))  g_sink += mpz_jacobi(p->a, p->modulus);
    else if (!strcmp(op, "isprime")) g_sink += mpz_probab_prime_p(p->a, p->prime_reps);
    else { fprintf(stderr, "unknown op: %s\n", op); exit(2); }
}

static const char *OPS[] = {"add",    "sub",    "mul",    "sqr",   "divrem",
                            "modulo", "modmul", "modpow", "gcd",   "gcdext",
                            "modinv", "jacobi", "isprime"};
static const unsigned long SIZES[] = {256, 1024, 2048, 4096};

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: pilot_gmp <op> | --list\n");
        return 2;
    }
    if (!strcmp(argv[1], "--list")) {
        for (size_t i = 0; i < sizeof(OPS) / sizeof(*OPS); i++)
            for (size_t j = 0; j < sizeof(SIZES) / sizeof(*SIZES); j++)
                printf("%s_%lu\n", OPS[i], SIZES[j]);
        return 0;
    }

    /* Split "<op>_<size>". */
    char name[64];
    strncpy(name, argv[1], sizeof(name) - 1);
    name[sizeof(name) - 1] = 0;
    char *us = strrchr(name, '_');
    if (!us) { fprintf(stderr, "bad op: %s\n", name); return 2; }
    *us = 0;
    unsigned long bits = strtoul(us + 1, NULL, 10);

    Pool p;
    pool_init(&p, bits);

    /* Self-calibrate reps to ~2 ms, then report per-op ms. */
    long reps = 1;
    double ms;
    for (;;) {
        double t0 = now_s();
        for (long i = 0; i < reps; i++) run(name, &p);
        double el = now_s() - t0;
        if (el >= 2e-3 || reps >= (1L << 26)) { ms = el * 1e3 / (double)reps; break; }
        reps *= 2;
    }
    printf("%.9f\n", ms);
    return 0;
}
