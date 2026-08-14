//! Microbenchmark harness for rump's multiprecision primitives.
//!
//!   `pilot_mp <op>`    draw ONE fresh random operand, repeat the primitive
//!                      until the elapsed interval exceeds the 2 ms
//!                      calibration floor, print the per-operation time in ms
//!   `pilot_mp --list`  every operation name
//!
//! Each invocation seeds its operands from the OS clock + pid, so it is a
//! *fresh* random trial. pilot-bench runs the program until the mean's
//! confidence interval converges, and its saved `readings.csv` is then a
//! sample of the random-input timing distribution: the mean is the average
//! cost, and the order statistics (min / p50 / p99 / max) are the
//! data-dependent extrema of a variable-time primitive. See
//! `scripts/bench_primitives.sh`, which drives both out of one run.
//!
//! Operation names are `<primitive>_<size>`, e.g. `mul_2048`, `divrem_2048`,
//! `montpow_rand_2048`, `jacobi_1024`, `gf2m_inv_571`.

use std::hint::black_box;
use std::time::{Duration, Instant};

use rump::{
    gcd, gcd_extended, is_probable_prime, jacobi, mod_inverse, mod_pow, sqrt_mod, BigUint, Gf2m,
    MontgomeryCtx,
};

// ─── Random operand generation ──────────────────────────────────────────────

/// splitmix64, seeded per process from the OS clock so every invocation is an
/// independent random trial.
struct SplitMix64(u64);

impl SplitMix64 {
    fn from_os() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Self(nanos ^ (u64::from(std::process::id())).wrapping_mul(0x9e37_79b9_7f4a_7c15))
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn biguint(&mut self, bits: usize) -> BigUint {
        let mut bytes = vec![0u8; bits.div_ceil(8)];
        for chunk in bytes.chunks_mut(8) {
            let word = self.next().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
        // Force the top bit so the value really is `bits` wide.
        let top = (bits - 1) % 8;
        bytes[0] &= (1u8 << top) - 1;
        bytes[0] |= 1u8 << top;
        BigUint::from_be_bytes(&bytes)
    }

    fn odd(&mut self, bits: usize) -> BigUint {
        let mut v = self.biguint(bits);
        if !v.is_odd() {
            v = v.add_ref(&BigUint::one());
        }
        v
    }
}

/// One random operand set for an integer size, plus the structures the modular
/// ops need. Built once per process, outside the timed region.
struct IntPool {
    a: BigUint,
    b: BigUint,
    /// Half-width divisor, so `div_rem` exercises the quotient loop rather than
    /// its `self < divisor` early return.
    divisor: BigUint,
    modulus: BigUint,
    /// Montgomery context and in-domain operands, built only for the ops that
    /// read them: the context setup is a full-width division, which at the
    /// scaling-sweep sizes (up to a million bits) would swamp a cheap op's
    /// measurement with pool construction.
    mont: Option<MontState>,
    /// A guaranteed quadratic residue modulo the pool's prime, for the
    /// residue-class-conditioned square-root rows.
    residue: Option<BigUint>,
    /// A random probable prime, built only for the ops that read one
    /// (`sqrtmod`, `isprime_true`): prime density thins as sizes grow, and
    /// hunting one beyond a few thousand bits costs seconds per trial —
    /// unpayable for every other op.
    prime: Option<BigUint>,
    /// Exponent pinned at 2^16 + 1 — two set bits, so the ladder is sixteen
    /// squarings and one multiply. This is the exponent floor: the row
    /// measures the kernel's scaling with the exponent's contribution at its
    /// minimum. The value is RSA's classical public exponent; the rationale
    /// is the exponent floor, not the workload.
    e65537: BigUint,
    /// Fresh random exponent of pinned 256-bit length — the realistic-exponent
    /// axis. Exponentiation costs (exponent bits) × (kernel cost), so holding
    /// the length constant keeps the size sweep a one-variable fit, and the
    /// linearity makes every other length derivable from this row.
    exp_rand: BigUint,
    /// Long-lived output storage for the in-place ops (`add`, `sub`),
    /// mirroring the GMP pilot's reused result `mpz_t`: after the first few
    /// calls its capacity covers every result, and the measurement is the
    /// arithmetic alone rather than an allocation per call.
    out: BigUint,
}

/// The Montgomery-domain slice of the pool.
struct MontState {
    ctx: MontgomeryCtx,
    a_mont: BigUint,
    b_mont: BigUint,
}

impl IntPool {
    fn mont(&self) -> &MontState {
        self.mont.as_ref().expect("op reads the Montgomery pool")
    }

    fn prime(&self) -> &BigUint {
        self.prime.as_ref().expect("op reads the prime")
    }
}

impl IntPool {
    fn new(bits: usize, op: &str) -> Self {
        let mut rng = SplitMix64::from_os();
        let a = rng.biguint(bits);
        let b = rng.biguint(bits);
        let divisor = rng.odd(bits / 2);
        let modulus = rng.odd(bits);
        let mont = matches!(
            op,
            "montmul" | "montsqr" | "montpow_e65537" | "montpow_rand"
        )
        .then(|| {
            let ctx = MontgomeryCtx::new(&modulus).expect("odd modulus");
            let a_mont = ctx.encode(&a);
            let b_mont = ctx.encode(&b);
            MontState { ctx, a_mont, b_mont }
        });
        // A prime near a fresh random point, for the prime-conditioned ops.
        // The residue-class-conditioned square-root rows pin the prime's
        // class — p ≡ 3 (mod 4) takes the (p+1)/4 shortcut, p ≡ 1 (mod 4)
        // the Tonelli–Shanks descent — decomposing the *residue half* of
        // the mixture. The unconditioned row's third and largest
        // population, non-residue rejection at the Jacobi test, stays in
        // that row: no weighted average of the conditioned rows
        // reconstructs it. The descent row's own spread is the geometric
        // s = v₂(p−1) tail (P(s = k | p ≡ 1 mod 4) = 2^(1−k) for k ≥ 2),
        // a property of the prime, not of the operand. The class filter
        // halves candidate density, so these rows' prime hunts cost about
        // twice the unconditioned row's per reading.
        let prime = matches!(
            op,
            "sqrtmod" | "sqrtmod_blum" | "sqrtmod_descent" | "isprime_true"
        )
        .then(|| {
            let wanted_mod4 = match op {
                "sqrtmod_blum" => Some(3),
                "sqrtmod_descent" => Some(1),
                _ => None,
            };
            let mut candidate = rng.odd(bits);
            loop {
                let class_ok = wanted_mod4.is_none_or(|r| candidate.rem_u64(4) == r);
                if class_ok && is_probable_prime(&candidate) {
                    break candidate;
                }
                candidate = candidate.add_ref(&BigUint::from_u64(2));
            }
        });
        // A guaranteed quadratic residue for the conditioned square-root
        // rows, so they measure root extraction rather than the Jacobi
        // rejection exit.
        let residue = matches!(op, "sqrtmod_blum" | "sqrtmod_descent").then(|| {
            let p = prime.as_ref().expect("conditioned ops build a prime");
            BigUint::mod_mul(&a, &a, p)
        });
        // Reused output storage for the ops that write in place; seeded from
        // `a` so its capacity already covers a full-width result.
        let out = a.clone();
        Self {
            a,
            b,
            divisor,
            modulus,
            mont,
            prime,
            residue,
            out,
            e65537: BigUint::from_u64(65_537),
            exp_rand: rng.biguint(256),
        }
    }
}

/// One random operand set for a binary field.
struct FieldPool {
    field: Gf2m,
    a: BigUint,
    b: BigUint,
    exp: BigUint,
}

impl FieldPool {
    fn new(degree: usize, taps: &[usize]) -> Self {
        let mut poly = BigUint::zero();
        poly.set_bit(degree);
        for &t in taps {
            poly.set_bit(t);
        }
        let field = Gf2m::new(poly).expect("irreducible field polynomial");
        let mut rng = SplitMix64::from_os();
        let reduce = |rng: &mut SplitMix64| field.mul(&rng.biguint(degree), &BigUint::one());
        let a = reduce(&mut rng);
        let b = reduce(&mut rng);
        Self {
            field,
            a,
            b,
            exp: rng.biguint(degree),
        }
    }
}

// ─── The operations ─────────────────────────────────────────────────────────

/// A benchmark closure over the process's single random operand set.
// One Bench exists per process, so the variants' size disparity buys
// nothing to fix and costs a pointer chase to "improve".
#[allow(clippy::large_enum_variant)]
enum Bench {
    Int(IntPool, fn(&mut IntPool)),
    Field(FieldPool, fn(&FieldPool)),
}

impl Bench {
    fn run(&mut self) {
        match self {
            Bench::Int(p, f) => f(p),
            Bench::Field(p, f) => f(p),
        }
    }
}

fn int_op(name: &str) -> Option<fn(&mut IntPool)> {
    Some(match name {
        // The two cheapest operations write into the pool's reused output
        // buffer — the same shape as the GMP pilot's `mpz_add(r, a, b)` into
        // a long-lived `r` — so both columns measure the arithmetic, not one
        // side's allocator.
        "add" => |p| {
            let IntPool { out, a, b, .. } = p;
            out.assign_add(a, b);
            black_box(out);
        },
        "sub" => |p| {
            let IntPool { out, a, b, .. } = p;
            let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
            out.assign_sub(hi, lo);
            black_box(out);
        },
        "mul" => |p| {
            black_box(p.a.mul_ref(&p.b));
        },
        "sqr" => |p| {
            black_box(p.a.square_ref());
        },
        "divrem" => |p| {
            black_box(p.a.div_rem(&p.divisor));
        },
        "modulo" => |p| {
            black_box(p.a.modulo(&p.divisor));
        },
        "modmul" => |p| {
            black_box(BigUint::mod_mul(&p.a, &p.b, &p.modulus));
        },
        "montsetup" => |p| {
            black_box(MontgomeryCtx::new(&p.modulus));
        },
        "montmul" => |p| {
            let m = p.mont();
            black_box(m.ctx.mul_mont(&m.a_mont, &m.b_mont));
        },
        "montsqr" => |p| {
            let m = p.mont();
            black_box(m.ctx.square_mont(&m.a_mont));
        },
        "montpow_e65537" => |p| {
            black_box(p.mont().ctx.pow(&p.a, &p.e65537));
        },
        "montpow_rand" => |p| {
            black_box(p.mont().ctx.pow(&p.a, &p.exp_rand));
        },
        "modpow" => |p| {
            black_box(mod_pow(&p.a, &p.exp_rand, &p.modulus));
        },
        "gcd" => |p| {
            black_box(gcd(&p.a, &p.b));
        },
        "gcdext" => |p| {
            black_box(gcd_extended(&p.a, &p.b));
        },
        "modinv" => |p| {
            black_box(mod_inverse(&p.a, &p.modulus));
        },
        "jacobi" => |p| {
            black_box(jacobi(&p.a, &p.modulus));
        },
        "sqrtmod" => |p| {
            black_box(sqrt_mod(&p.a, p.prime()));
        },
        "sqrtmod_blum" | "sqrtmod_descent" => |p| {
            let residue = p.residue.as_ref().expect("op builds a residue");
            black_box(sqrt_mod(residue, p.prime()));
        },
        "isprime" => |p| {
            black_box(is_probable_prime(&p.a));
        },
        "isprime_true" => |p| {
            // The outcome-conditioned cost: a prime pays the sieve plus all
            // twelve Miller-Rabin rounds. On a fully random operand the mean
            // is dominated by trivial rejections; this row measures the cost
            // a caller plans around.
            black_box(is_probable_prime(p.prime()));
        },
        _ => return None,
    })
}

fn field_op(name: &str) -> Option<fn(&FieldPool)> {
    Some(match name {
        "gf2m_mul" => |p| {
            black_box(p.field.mul(&p.a, &p.b));
        },
        "gf2m_sqr" => |p| {
            black_box(p.field.square(&p.a));
        },
        "gf2m_inv" => |p| {
            black_box(p.field.inverse(&p.a));
        },
        "gf2m_pow" => |p| {
            black_box(p.field.pow(&p.a, &p.exp));
        },
        "gf2m_sqrt" => |p| {
            black_box(p.field.sqrt(&p.a));
        },
        _ => return None,
    })
}

const INT_SIZES: &[usize] = &[256, 1024, 2048, 4096];
const INT_OPS: &[&str] = &[
    "add",
    "sub",
    "mul",
    "sqr",
    "divrem",
    "modulo",
    "modmul",
    "montsetup",
    "montmul",
    "montsqr",
    "montpow_e65537",
    "montpow_rand",
    "modpow",
    "gcd",
    "gcdext",
    "modinv",
    "jacobi",
    "sqrtmod",
    "sqrtmod_blum",
    "sqrtmod_descent",
    "isprime",
    "isprime_true",
];

/// (degree, taps) for two representative FIPS binary fields.
const FIELDS: &[(usize, &[usize])] = &[(233, &[74, 0]), (571, &[10, 5, 2, 0])];
const FIELD_OPS: &[&str] = &["gf2m_mul", "gf2m_sqr", "gf2m_inv", "gf2m_pow", "gf2m_sqrt"];

fn build(full: &str) -> Option<Bench> {
    let (name, num) = full.rsplit_once('_')?;
    let num: usize = num.parse().ok()?;
    if let Some(f) = int_op(name) {
        return Some(Bench::Int(IntPool::new(num, name), f));
    }
    if let Some(f) = field_op(name) {
        let (_, taps) = FIELDS.iter().find(|(d, _)| *d == num)?;
        return Some(Bench::Field(FieldPool::new(num, taps), f));
    }
    None
}

fn all_ops() -> Vec<String> {
    let mut out = Vec::new();
    for &size in INT_SIZES {
        for &op in INT_OPS {
            out.push(format!("{op}_{size}"));
        }
    }
    for &(deg, _) in FIELDS {
        for &op in FIELD_OPS {
            out.push(format!("{op}_{deg}"));
        }
    }
    out
}

// ─── One reading ────────────────────────────────────────────────────────────

/// Repeat the op on its single random operand until the elapsed interval
/// exceeds the calibration floor,
/// then print the per-op cost in ms. The whole batch uses the *same* operand,
/// so the reading reflects that operand's data-dependent cost; the fresh draw
/// per process is what makes the collection of readings a random sample.
fn one_reading(bench: &mut Bench) {
    let target = Duration::from_millis(2);
    let mut reps = 1usize;
    let ms = loop {
        let start = Instant::now();
        for _ in 0..reps {
            bench.run();
        }
        let elapsed = start.elapsed();
        if elapsed >= target || reps >= 1 << 26 {
            break elapsed.as_secs_f64() * 1e3 / reps as f64;
        }
        reps *= 2;
    };
    println!("{ms:.9}");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--list") => {
            for op in all_ops() {
                println!("{op}");
            }
        }
        Some(name) => {
            let mut bench = build(name).unwrap_or_else(|| panic!("unknown op: {name}"));
            one_reading(&mut bench);
        }
        None => {
            eprintln!("usage: pilot_mp <op> | --list");
            std::process::exit(2);
        }
    }
}
