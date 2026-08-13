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
    /// A random probable prime, built only for `sqrtmod`: prime density thins
    /// as sizes grow, and hunting one beyond a few thousand bits costs minutes
    /// — unpayable per pilot trial, and pointless for every other op.
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
        // A prime near a fresh random point, for sqrt_mod.
        let prime = (op == "sqrtmod").then(|| {
            let mut candidate = rng.odd(bits);
            while !is_probable_prime(&candidate) {
                candidate = candidate.add_ref(&BigUint::from_u64(2));
            }
            candidate
        });
        Self {
            a,
            b,
            divisor,
            modulus,
            mont,
            prime,
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
enum Bench {
    Int(IntPool, fn(&IntPool)),
    Field(FieldPool, fn(&FieldPool)),
}

impl Bench {
    fn run(&self) {
        match self {
            Bench::Int(p, f) => f(p),
            Bench::Field(p, f) => f(p),
        }
    }
}

fn int_op(name: &str) -> Option<fn(&IntPool)> {
    Some(match name {
        "add" => |p| {
            black_box(p.a.add_ref(&p.b));
        },
        "sub" => |p| {
            let (hi, lo) = if p.a >= p.b {
                (&p.a, &p.b)
            } else {
                (&p.b, &p.a)
            };
            black_box(hi.sub_ref(lo));
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
        "isprime" => |p| {
            black_box(is_probable_prime(&p.a));
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
    "isprime",
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
fn one_reading(bench: &Bench) {
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
            let bench = build(name).unwrap_or_else(|| panic!("unknown op: {name}"));
            one_reading(&bench);
        }
        None => {
            eprintln!("usage: pilot_mp <op> | --list");
            std::process::exit(2);
        }
    }
}
