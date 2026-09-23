//! Ad-hoc bigint microbenchmarks.
//!
//! Usage:
//!   cargo run --release --bin bench_bigint
//!   cargo run --release --bin bench_bigint -- 256 512 1024
//!
//! Prints Markdown with ns/op for core bigint kernels.

use std::hint::black_box;
use std::time::{Duration, Instant};

use rump::modular::MontgomeryContext;
use rump::BigUint;

/// Deterministic operand generator: splitmix64 (Steele, Lea & Flood 2014),
/// vendored so the benchmark needs no dependency. Not a CSPRNG; the
/// benchmark needs only reproducible, well-scattered operand draws.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn fill_bytes(&mut self, out: &mut [u8]) {
        for chunk in out.chunks_mut(8) {
            let word = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
    }
}

/// Operand widths in bits when none are given: doublings from 256 to
/// 4096, the span of RSA and Diffie–Hellman modulus sizes.
const DEFAULT_BITS: &[usize] = &[256, 512, 1024, 2048, 4096];
/// Time budget per row: iterations double until one batch lasts this long
/// or reaches `MAX_ITERS`, so a row averages over at least that much work.
const TARGET: Duration = Duration::from_millis(200);
/// Cap on the doubling, so a cheap kernel is reported after 2^16 calls
/// rather than the millions the budget would demand of it.
const MAX_ITERS: usize = 1 << 16;
/// First batch sizes for the doubling. A product is fast enough that a
/// batch of one is a wasted round; an exponentiation or a division is not.
const FAST_START_ITERS: usize = 2;
const SLOW_START_ITERS: usize = 1;
/// Arbitrary, fixed so every run draws the same operands.
const SEED: u64 = 0x4d4d_4d4d_4d4d_4d4d;
/// 2¹⁶ + 1, RSA's classical public exponent: two set bits, so the ladder
/// is sixteen squarings and one multiply, the exponent's floor.
const F4: u64 = 65_537;
/// Random exponent length. Exponentiation costs (exponent bits) times the
/// kernel, so a fixed length keeps the width sweep a one-variable fit.
const RANDOM_EXPONENT_BITS: usize = 256;

#[derive(Clone, Debug)]
struct Config {
    bits: Vec<usize>,
    repeat: usize,
}

fn parse_config() -> Config {
    let mut out = Vec::new();
    let mut repeat = 1usize;
    let mut args = std::env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        if arg == "--repeat" {
            if let Some(value) = args.next() {
                if let Ok(parsed) = value.parse::<usize>() {
                    repeat = parsed.max(1);
                }
            }
            continue;
        }
        if let Ok(bits) = arg.parse::<usize>() {
            out.push(bits);
        }
    }
    let bits = if out.is_empty() {
        DEFAULT_BITS.to_vec()
    } else {
        out
    };
    Config { bits, repeat }
}

fn random_biguint(rng: &mut SplitMix64, bits: usize) -> BigUint {
    let bits = bits.max(1);
    let byte_len = bits.div_ceil(8);
    let mut bytes = vec![0u8; byte_len.max(1)];
    rng.fill_bytes(&mut bytes);

    // Clear the bits above the requested width before forcing the top one, so
    // the result is exactly `bits` bits wide when `bits` is not a multiple
    // of 8.
    let top_bit = (bits - 1) % 8;
    bytes[0] &= (1u8 << top_bit) - 1;
    bytes[0] |= 1u8 << top_bit;
    BigUint::from_be_bytes(&bytes)
}

fn random_odd_biguint(rng: &mut SplitMix64, bits: usize) -> BigUint {
    let mut value = random_biguint(rng, bits);
    if !value.is_odd() {
        value = value.add(&BigUint::one());
    }
    value
}

fn bench_ns_per_op(mut op: impl FnMut(), min_iters: usize) -> (usize, f64) {
    let mut iters = min_iters.max(1);
    loop {
        let start = Instant::now();
        for _ in 0..iters {
            op();
        }
        let elapsed = start.elapsed();
        if elapsed >= TARGET || iters >= MAX_ITERS {
            let ns = elapsed.as_secs_f64() * 1e9 / iters as f64;
            return (iters, ns);
        }
        iters = (iters * 2).min(MAX_ITERS);
    }
}

fn run_for_bits(rng: &mut SplitMix64, bits: usize) {
    let lhs = random_biguint(rng, bits);
    let rhs = random_biguint(rng, bits);
    let modulus = random_odd_biguint(rng, bits);
    let base = random_biguint(rng, bits);
    // Half-width divisor: a full-width one leaves a quotient of a bit or two,
    // so the timing says almost nothing about the quotient loop, and whenever
    // the draw lands above the dividend `div_rem` returns through its
    // `self < divisor` early exit and times a clone instead of a division.
    let divisor = random_biguint(rng, bits / 2).add(&BigUint::one());
    assert!(
        lhs >= divisor,
        "divisor must not trip the div_rem fast path"
    );
    let e_65537 = BigUint::from_u64(F4);
    let exp_random = random_biguint(rng, RANDOM_EXPONENT_BITS);
    let ctx = MontgomeryContext::new(&modulus).expect("odd modulus");

    println!("\n### {}-bit", bits);
    println!("| Operation | ns/op | Iters |");
    println!("|-----------|------:|------:|");

    let (iters, ns) = bench_ns_per_op(
        || {
            black_box(lhs.mul(&rhs));
        },
        FAST_START_ITERS,
    );
    println!("| mul | {:.1} | {} |", ns, iters);

    let (iters, ns) = bench_ns_per_op(
        || {
            black_box(BigUint::mod_mul(&lhs, &rhs, &modulus));
        },
        FAST_START_ITERS,
    );
    println!("| mod_mul (odd modulus) | {:.1} | {} |", ns, iters);

    let (iters, ns) = bench_ns_per_op(
        || {
            black_box(ctx.pow(&base, &e_65537));
        },
        SLOW_START_ITERS,
    );
    println!("| montgomery_pow (e=65537) | {:.1} | {} |", ns, iters);

    let (iters, ns) = bench_ns_per_op(
        || {
            black_box(ctx.pow(&base, &exp_random));
        },
        SLOW_START_ITERS,
    );
    println!("| montgomery_pow (random 256b e) | {:.1} | {} |", ns, iters);

    let (iters, ns) = bench_ns_per_op(
        || {
            black_box(lhs.div_rem(&divisor));
        },
        SLOW_START_ITERS,
    );
    println!("| div_rem | {:.1} | {} |", ns, iters);

    let (iters, ns) = bench_ns_per_op(
        || {
            black_box(lhs.rem(&divisor));
        },
        SLOW_START_ITERS,
    );
    println!("| rem | {:.1} | {} |", ns, iters);
}

fn main() {
    let cfg = parse_config();
    let mut rng = SplitMix64::new(SEED);

    println!("# Bigint Kernel Microbenchmarks");
    println!(
        "Columns: nanoseconds per operation and iterations used. (repeat={})",
        cfg.repeat
    );
    for pass in 0..cfg.repeat {
        if cfg.repeat > 1 {
            println!("\n## Pass {}", pass + 1);
        }
        for &bits in &cfg.bits {
            run_for_bits(&mut rng, bits);
        }
    }
}
