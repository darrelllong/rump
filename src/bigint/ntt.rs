//! Exact large-integer multiplication by number-theoretic transform.
//!
//! The limbs are the digits: a value is convolved as a polynomial in
//! `2^64` modulo three primes between `2^60` and `2^62`, and each coefficient of
//! the integer convolution is recovered from its three residues by the
//! Chinese remainder theorem (Garner, *The residue number system*, IRE
//! Trans. Electronic Computers 8 (1959), 140–147), then an ordinary carry
//! pass returns to limbs. This is the small-prime number-theoretic
//! transform (Pollard, *The Fast Fourier Transform in a Finite Field*,
//! Math. Comp. 25 (1971), 365–374), radix 2 after Cooley & Tukey (Math.
//! Comp. 19 (1965), 297–301), decimation in frequency forward and in time
//! back, so that the spectrum is never permuted. It is not Schönhage and
//! Strassen's recursion over `Z/(2^n + 1)`.
//!
//! A coefficient of the convolution of two `n`-digit values is at most
//! `n·(2^64 − 1)²`, and the product of the three primes is above `2^183`, so
//! coefficients are exact through transforms of `2^55` values; the primes'
//! 2-adic valuations allow as much. Each prime `p = c·2^k + 1` is above
//! `2^60`, which the reductions below rely on, and below `2^62`, so that a
//! sum of two residues and a Shoup remainder under `2p` fit a word.
//!
//! Modular products are by two forms, neither a division (Harvey, *Faster
//! arithmetic for number-theoretic transforms*, J. Symbolic Computation 60
//! (2014), 113–119, §2): a twiddle `w` carries `⌊w·2^64/p⌋`, and `x·w mod
//! p` is `x·w − ⌊x·⌊w·2^64/p⌋/2^64⌋·p`, which lies in `[0, 2p)`; a product
//! of two variables, `x·y < 2^124`, is reduced by Barrett's estimate
//! `⌊⌊xy/2^60⌋·⌊2^124/p⌋/2^64⌋` of the quotient, which is short by at most
//! two, so the remainder lies in `[0, 3p)` and two subtractions settle it
//! ([`reduce_wide`] derives the bound).
//!
//! Against the transform this replaces — digits of sixteen bits under two
//! primes below `2^31` — a product of the same width takes a transform a
//! quarter as long, nine transforms where there were six, and so a third
//! of the butterflies with twice the bytes each: half the traffic through
//! memory, which paced the old transform past the cache. The bound of
//! `2^26` values, `2^30` bits of product, is gone with the digits
//! (PERFORMANCE.md, "Digits of a limb").

use super::BigUint;

/// A transform is at most this long: the least 2-adic valuation among the
/// primes, `2^55`, which is also where `n·(2^64 − 1)²` would pass their
/// product (`transform_len_is_exact`); on a machine whose `usize` is
/// narrower, the largest power of two it holds. Memory bounds a transform
/// long before either.
const MAX_TRANSFORM_LEN: usize = if usize::BITS > 55 {
    1 << 55
} else {
    1 << (usize::BITS - 1)
};

// The primes `c·2^k + 1`, the largest three below `2^62` with `k ≥ 55`, each
// with its least primitive root, found by testing 2, 3, … against the
// factors of `p − 1` (`primes_and_roots_hold`).
const PRIME_0: u64 = 4_179_340_454_199_820_289; // 29·2^57 + 1
const ROOT_0: u64 = 3;
const PRIME_1: u64 = 2_485_986_994_308_513_793; // 69·2^55 + 1
const ROOT_1: u64 = 5;
const PRIME_2: u64 = 2_053_641_430_080_946_177; // 57·2^55 + 1
const ROOT_2: u64 = 7;

// Garner's coefficients: `p₀⁻¹ mod p₁` and `(p₀·p₁)⁻¹ mod p₂`
// (`garner_constants_hold`).
const INVERSE_P0_MOD_P1: u64 = 1_745_480_230_046_403_300;
const INVERSE_P0P1_MOD_P2: u64 = 1_296_578_699_500_258_396;
const P0P1: u128 = (PRIME_0 as u128) * (PRIME_1 as u128);

// Linear passes (the digits in and out, pointwise products, residue
// copies, the CRT) need enough values per context to repay one scoped
// worker wave. The grain is from the old kernel's `ntt_phase_profile` and
// `ntt_worker_scaling_timing` probes (PERFORMANCE.md, "Exact NTT
// multiplication"); a value is now a limb where it was a quarter of one.
const MIN_LINEAR_VALUES_PER_WORKER: usize = 1 << 16;

// Below 2^14 values the NTT loses to the recursive multiplication ladder
// even with four contexts. Keeping forced small-kernel tests serial also
// stops thread launch from dominating their work.
const PARALLEL_TRANSFORM_MIN_LEN: usize = 1 << 14;

// A stage's twiddles are a table, not a running power: a power carried
// through the loop is a dependency chain, and a stage ran at a
// multiplication's latency a butterfly. The table is at most 2^16 entries
// (a megabyte with their quotients, a core's second-level cache); a wider
// stage applies it under each span's leading power, one more independent
// multiplication a butterfly. Measured in PERFORMANCE.md ("A stage's
// twiddles as a table"). An entry costs a Barrett division and a Shoup
// quotient, some fifteen nanoseconds, against a nanosecond a butterfly, so
// a table is built once a process for its prime, direction and width and
// kept ([`stage_table`]): built a transform, the tables of a 2^16-value
// transform cost as much as its butterflies.
const TWIDDLE_TABLE_LEN: usize = 1 << 16;

// A transform's stages run depth first over blocks of this many values, so
// that a block's stages find it in cache: a stage at a time over the whole
// array reads and writes it once a stage, and past the cache the transform
// ran at memory's pace. A block of 2^13 values is 64 KB, inside a core's
// second-level cache with its twiddles (PERFORMANCE.md, "No permutation, no
// call a block").
const CACHE_BLOCK_LEN: usize = 1 << 13;

// Worker targets by transform length: each row is the longest transform it
// serves and the target for it; a longer transform takes
// MAX_WORKER_TARGET. The transform past the cache is paced by memory, and
// threads past the point where its bandwidth is spent add CPU and no speed:
// on deepcore's EPYC 7702 a 2^28-bit product (2^23 values) takes 3.08 s on
// one thread, 1.78 on two, 1.10 on four, 0.93 on eight, 0.55 on sixteen
// and 0.55 on thirty-two, its CPU 1.0, 1.1, 1.4, 1.8, 2.0 and 2.2 times the
// thread's; a 2^26-bit product (2^21 values) is fastest on eight. The
// targets below the measured lengths follow the old kernel's rows, a
// quarter the length for the same bytes (PERFORMANCE.md, "Digits of a
// limb").
const WORKER_TARGETS: [(usize, usize); 3] = [(1 << 14, 4), (1 << 16, 8), (1 << 21, 8)];
const MAX_WORKER_TARGET: usize = 16;

/// Padded transform length for full-width operands, when supported.
pub(super) fn transform_len(lhs_limbs: usize, rhs_limbs: usize) -> Option<usize> {
    let convolution_len = lhs_limbs.checked_add(rhs_limbs)?.checked_sub(1)?;
    convolution_len
        .checked_next_power_of_two()
        .filter(|&len| len <= MAX_TRANSFORM_LEN)
}

/// Number of execution contexts the automatic transform will actually use.
pub(super) fn automatic_worker_count(transform_len: usize) -> usize {
    let available = crate::parallel::budget();
    worker_count(transform_len, available)
}

/// Select a power-of-two worker count from a hard context ceiling.
///
/// The target comes from `WORKER_TARGETS`, is capped at `max_contexts`, and
/// is rounded down to a power of two, so a smaller machine uses only what
/// it reports.
pub(super) fn worker_count(transform_len: usize, max_contexts: usize) -> usize {
    debug_assert!(transform_len.is_power_of_two());
    if transform_len < PARALLEL_TRANSFORM_MIN_LEN || max_contexts <= 1 {
        return 1;
    }
    let target = WORKER_TARGETS
        .iter()
        .find(|&&(longest, _)| transform_len <= longest)
        .map_or(MAX_WORKER_TARGET, |&(_, target)| target);
    let maximum = max_contexts.min(target);
    1usize << maximum.ilog2()
}

/// Multiply two non-zero values through an exact three-prime NTT convolution.
pub(super) fn multiply(lhs: &BigUint, rhs: &BigUint) -> BigUint {
    let contexts = crate::parallel::budget();
    multiply_impl(lhs, rhs, contexts)
}

/// Square one non-zero value: one transform buffer a prime, pointwise
/// self-products, and the same CRT.
pub(super) fn square(value: &BigUint) -> BigUint {
    debug_assert!(!value.limbs.is_empty());
    let digits = value.limbs.len();
    let convolution_len = 2 * digits - 1;
    let transform_len = transform_len(digits, digits).expect("NTT square transform is supported");
    let workers = automatic_worker_count(transform_len);

    let mut values = vec![0u64; transform_len];
    let mut residues = Vec::with_capacity(2);
    for prime in 0..2 {
        write_digits_prime(prime, value, &mut values, workers);
        square_mod_prime(prime, &mut values, workers);
        residues.push(copy_residues(&values, convolution_len, workers));
        parallel_fill_zero(&mut values, workers);
    }
    write_digits_prime(2, value, &mut values, workers);
    square_mod_prime(2, &mut values, workers);
    reconstruct(
        &residues[0],
        &residues[1],
        &values,
        convolution_len,
        workers,
    )
}

/// Serial reference for differential and crossover measurement.
#[cfg(test)]
pub(super) fn multiply_serial(lhs: &BigUint, rhs: &BigUint) -> BigUint {
    multiply_impl(lhs, rhs, 1)
}

/// Forced context limit for parallel-scaling measurement.
#[cfg(test)]
pub(super) fn multiply_with_contexts(lhs: &BigUint, rhs: &BigUint, max_contexts: usize) -> BigUint {
    multiply_impl(lhs, rhs, max_contexts.max(1))
}

fn multiply_impl(lhs: &BigUint, rhs: &BigUint, max_contexts: usize) -> BigUint {
    multiply_impl_selecting_workers(lhs, rhs, |transform_len| {
        worker_count(transform_len, max_contexts)
    })
}

/// Forced exact worker count for scaling measurements.
#[cfg(test)]
pub(super) fn multiply_with_workers(lhs: &BigUint, rhs: &BigUint, workers: usize) -> BigUint {
    assert!(
        workers.is_power_of_two(),
        "NTT worker count must be a power of two"
    );
    multiply_impl_selecting_workers(lhs, rhs, |transform_len| workers.min(transform_len))
}

fn multiply_impl_selecting_workers(
    lhs: &BigUint,
    rhs: &BigUint,
    select_workers: impl FnOnce(usize) -> usize,
) -> BigUint {
    debug_assert!(!lhs.limbs.is_empty() && !rhs.limbs.is_empty());
    let convolution_len = lhs.limbs.len() + rhs.limbs.len() - 1;
    let transform_len =
        transform_len(lhs.limbs.len(), rhs.limbs.len()).expect("NTT transform is supported");
    let workers = select_workers(transform_len);
    debug_assert!(workers.is_power_of_two() && workers <= transform_len);

    let mut left = vec![0u64; transform_len];
    let mut right = vec![0u64; transform_len];
    let mut residues = Vec::with_capacity(2);
    for prime in 0..2 {
        write_input_pair_prime(prime, lhs, rhs, &mut left, &mut right, workers);
        convolve_mod_prime(prime, &mut left, &mut right, workers);
        residues.push(copy_residues(&left, convolution_len, workers));
        clear_pair(&mut left, &mut right, workers);
    }
    write_input_pair_prime(2, lhs, rhs, &mut left, &mut right, workers);
    convolve_mod_prime(2, &mut left, &mut right, workers);
    reconstruct(&residues[0], &residues[1], &left, convolution_len, workers)
}

/// The convolution under prime `prime` of the three, left in `left`.
fn convolve_mod_prime(prime: usize, left: &mut [u64], right: &mut [u64], workers: usize) {
    match prime {
        0 => convolve_mod::<PRIME_0, ROOT_0>(left, right, workers),
        1 => convolve_mod::<PRIME_1, ROOT_1>(left, right, workers),
        _ => convolve_mod::<PRIME_2, ROOT_2>(left, right, workers),
    }
}

fn square_mod_prime(prime: usize, values: &mut [u64], workers: usize) {
    match prime {
        0 => square_mod::<PRIME_0, ROOT_0>(values, workers),
        1 => square_mod::<PRIME_1, ROOT_1>(values, workers),
        _ => square_mod::<PRIME_2, ROOT_2>(values, workers),
    }
}

/// Wall-clock phase breakdown for the ignored NTT profiling probe, summed
/// over the three primes.
#[cfg(test)]
pub(super) struct MultiplicationProfile {
    pub(super) allocate: std::time::Duration,
    pub(super) prepare_inputs: std::time::Duration,
    pub(super) forward: std::time::Duration,
    pub(super) pointwise: std::time::Duration,
    pub(super) inverse: std::time::Duration,
    pub(super) residue_copy: std::time::Duration,
    pub(super) clear: std::time::Duration,
    pub(super) reconstruct: std::time::Duration,
}

/// Multiply while timing the high-level phases. Test builds only, so
/// production calls pay for no timestamps.
#[cfg(test)]
pub(super) fn multiply_profiled(
    lhs: &BigUint,
    rhs: &BigUint,
    workers: usize,
) -> (BigUint, MultiplicationProfile) {
    use std::time::{Duration, Instant};

    assert!(
        workers.is_power_of_two(),
        "NTT worker count must be a power of two"
    );
    let convolution_len = lhs.limbs.len() + rhs.limbs.len() - 1;
    let transform_len = convolution_len.next_power_of_two();
    assert!(transform_len <= MAX_TRANSFORM_LEN);
    let workers = workers.min(transform_len);

    let started = Instant::now();
    let mut left = vec![0u64; transform_len];
    let mut right = vec![0u64; transform_len];
    let allocate = started.elapsed();
    let (mut prepare_inputs, mut forward, mut pointwise, mut inverse, mut residue_copy, mut clear) = (
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
    );
    let mut residues = Vec::with_capacity(2);
    for prime in 0..3 {
        let started = Instant::now();
        write_input_pair_prime(prime, lhs, rhs, &mut left, &mut right, workers);
        prepare_inputs += started.elapsed();
        let phases = match prime {
            0 => convolve_mod_timed::<PRIME_0, ROOT_0>(&mut left, &mut right, workers),
            1 => convolve_mod_timed::<PRIME_1, ROOT_1>(&mut left, &mut right, workers),
            _ => convolve_mod_timed::<PRIME_2, ROOT_2>(&mut left, &mut right, workers),
        };
        forward += phases.0;
        pointwise += phases.1;
        inverse += phases.2;
        if prime < 2 {
            let started = Instant::now();
            residues.push(copy_residues(&left, convolution_len, workers));
            residue_copy += started.elapsed();
            let started = Instant::now();
            clear_pair(&mut left, &mut right, workers);
            clear += started.elapsed();
        }
    }
    let started = Instant::now();
    let product = reconstruct(&residues[0], &residues[1], &left, convolution_len, workers);
    let reconstruct = started.elapsed();
    (
        product,
        MultiplicationProfile {
            allocate,
            prepare_inputs,
            forward,
            pointwise,
            inverse,
            residue_copy,
            clear,
            reconstruct,
        },
    )
}

#[cfg(test)]
fn convolve_mod_timed<const MODULUS: u64, const ROOT: u64>(
    left: &mut [u64],
    right: &mut [u64],
    workers: usize,
) -> (
    std::time::Duration,
    std::time::Duration,
    std::time::Duration,
) {
    use std::time::Instant;
    let started = Instant::now();
    forward_pair::<MODULUS, ROOT>(left, right, workers);
    let forward = started.elapsed();
    let started = Instant::now();
    pointwise_multiply::<MODULUS>(left, right, workers);
    let pointwise = started.elapsed();
    let started = Instant::now();
    transform_from_bit_reversed::<MODULUS, ROOT>(left, true, workers);
    (forward, pointwise, started.elapsed())
}

/// Coefficients from their residues, carried into limbs.
fn reconstruct(
    residues_0: &[u64],
    residues_1: &[u64],
    residues_2: &[u64],
    convolution_len: usize,
    workers: usize,
) -> BigUint {
    debug_assert_eq!(residues_0.len(), convolution_len);
    debug_assert_eq!(residues_1.len(), convolution_len);
    debug_assert!(residues_2.len() >= convolution_len);
    let mut coefficients = vec![[0u64; 3]; convolution_len];
    let workers = linear_worker_count(convolution_len, workers);
    let chunk_len = convolution_len.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut chunks = residues_0
            .chunks(chunk_len)
            .zip(residues_1.chunks(chunk_len))
            .zip(residues_2[..convolution_len].chunks(chunk_len))
            .zip(coefficients.chunks_mut(chunk_len));
        let (((caller_0, caller_1), caller_2), caller_out) = chunks
            .next()
            .expect("an NTT convolution has at least one coefficient");
        for (((r0, r1), r2), out) in chunks {
            let _ = scope.spawn(move || reconstruct_chunk(r0, r1, r2, out));
        }
        reconstruct_chunk(caller_0, caller_1, caller_2, caller_out);
    });
    // The carry: coefficient `j` is added at limb `j`. A coefficient is
    // under `2^184` and a carry under `2^121`, so the running sum fits a
    // `u128` and its high limb a `u128` again.
    let mut limbs = Vec::with_capacity(convolution_len + 2);
    let mut carry = 0u128;
    for coefficient in &coefficients {
        let low = u128::from(coefficient[0]) + carry;
        limbs.push(low as u64);
        carry = (low >> 64) + u128::from(coefficient[1]) + (u128::from(coefficient[2]) << 64);
    }
    while carry != 0 {
        limbs.push(carry as u64);
        carry >>= 64;
    }
    BigUint::from_limbs(limbs)
}

fn reconstruct_chunk(r0: &[u64], r1: &[u64], r2: &[u64], out: &mut [[u64; 3]]) {
    for (((&r0, &r1), &r2), coefficient) in r0.iter().zip(r1).zip(r2).zip(out) {
        *coefficient = crt_three(r0, r1, r2);
    }
}

/// The `x < p₀·p₁·p₂` with the given residues, as three limbs, by Garner's
/// mixed radix: `x = r₀ + p₀·t₁ + p₀p₁·t₂` with `t₁ = (r₁ − r₀)·p₀⁻¹ mod p₁`
/// and `t₂ = (r₂ − (r₀ + p₀t₁))·(p₀p₁)⁻¹ mod p₂`.
fn crt_three(r0: u64, r1: u64, r2: u64) -> [u64; 3] {
    let t1 = mul_mod::<PRIME_1>(
        sub_mod::<PRIME_1>(r1, reduce_word::<PRIME_1>(r0)),
        INVERSE_P0_MOD_P1,
    );
    // r₀ + p₀·t₁ < p₀·p₁ < 2^123.
    let y = u128::from(r0) + u128::from(PRIME_0) * u128::from(t1);
    let t2 = mul_mod::<PRIME_2>(
        sub_mod::<PRIME_2>(r2, reduce_wide::<PRIME_2>(y)),
        INVERSE_P0P1_MOD_P2,
    );
    // p₀p₁·t₂ in two pieces, each under 2^125.
    let low = (P0P1 as u64 as u128) * u128::from(t2);
    let high = (P0P1 >> 64) * u128::from(t2);
    let sum_low = y + low;
    let sum_high = (sum_low >> 64) + high;
    [sum_low as u64, sum_high as u64, (sum_high >> 64) as u64]
}

/// `a − b mod p` for `a, b < p`.
#[inline]
fn sub_mod<const MODULUS: u64>(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        a + MODULUS - b
    }
}

/// `x mod p` for a word `x`: a division by a constant, which compiles to a
/// multiplication by its reciprocal.
#[inline]
fn reduce_word<const MODULUS: u64>(x: u64) -> u64 {
    x % MODULUS
}

/// A value's limbs, each reduced modulo the prime, into the first places of
/// a zero-filled transform input, in natural order, a run a worker.
fn write_digits<const MODULUS: u64>(value: &BigUint, digits: &mut [u64], workers: usize) {
    let len = value.limbs.len();
    debug_assert!(digits.len() >= len);
    let digits = &mut digits[..len];
    let workers = linear_worker_count(len.max(1), workers);
    let chunk_len = len.div_ceil(workers).max(1);
    std::thread::scope(|scope| {
        let mut chunks = digits
            .chunks_mut(chunk_len)
            .zip(value.limbs.chunks(chunk_len));
        let Some((caller_chunk, caller_source)) = chunks.next() else {
            return;
        };
        for (chunk, source) in chunks {
            let _ = scope.spawn(move || write_digit_run::<MODULUS>(chunk, source));
        }
        write_digit_run::<MODULUS>(caller_chunk, caller_source);
    });
}

fn write_digit_run<const MODULUS: u64>(digits: &mut [u64], limbs: &[u64]) {
    for (digit, &limb) in digits.iter_mut().zip(limbs) {
        *digit = reduce_word::<MODULUS>(limb);
    }
}

fn write_input_pair<const MODULUS: u64>(
    lhs: &BigUint,
    rhs: &BigUint,
    left: &mut [u64],
    right: &mut [u64],
    workers: usize,
) {
    debug_assert_eq!(left.len(), right.len());
    if workers == 1 {
        write_digits::<MODULUS>(lhs, left, 1);
        write_digits::<MODULUS>(rhs, right, 1);
        return;
    }
    let input_workers = workers / 2;
    std::thread::scope(|scope| {
        let _ = scope.spawn(|| write_digits::<MODULUS>(lhs, left, input_workers));
        write_digits::<MODULUS>(rhs, right, input_workers);
    });
}

/// The inputs reduced under prime `prime` of the three.
fn write_input_pair_prime(
    prime: usize,
    lhs: &BigUint,
    rhs: &BigUint,
    left: &mut [u64],
    right: &mut [u64],
    workers: usize,
) {
    match prime {
        0 => write_input_pair::<PRIME_0>(lhs, rhs, left, right, workers),
        1 => write_input_pair::<PRIME_1>(lhs, rhs, left, right, workers),
        _ => write_input_pair::<PRIME_2>(lhs, rhs, left, right, workers),
    }
}

fn write_digits_prime(prime: usize, value: &BigUint, digits: &mut [u64], workers: usize) {
    match prime {
        0 => write_digits::<PRIME_0>(value, digits, workers),
        1 => write_digits::<PRIME_1>(value, digits, workers),
        _ => write_digits::<PRIME_2>(value, digits, workers),
    }
}

/// The cyclic convolution of `left` and `right` modulo `MODULUS`, left in
/// `left` in natural order; `right` is spent.
fn convolve_mod<const MODULUS: u64, const ROOT: u64>(
    left: &mut [u64],
    right: &mut [u64],
    workers: usize,
) {
    debug_assert_eq!(left.len(), right.len());
    forward_pair::<MODULUS, ROOT>(left, right, workers);
    pointwise_multiply::<MODULUS>(left, right, workers);
    transform_from_bit_reversed::<MODULUS, ROOT>(left, true, workers);
}

/// Both forward transforms, at once where there are workers to share.
fn forward_pair<const MODULUS: u64, const ROOT: u64>(
    left: &mut [u64],
    right: &mut [u64],
    workers: usize,
) {
    if workers == 1 {
        forward_dif::<MODULUS, ROOT>(left, 1);
        forward_dif::<MODULUS, ROOT>(right, 1);
    } else {
        // The two forward transforms are independent. Running both at once
        // with half the worker budget removes one whole transform from the
        // critical path while keeping the total live contexts at `workers`.
        // The inverse has no independent peer and uses the full budget.
        let forward_workers = workers / 2;
        std::thread::scope(|scope| {
            let _ = scope.spawn(|| forward_dif::<MODULUS, ROOT>(left, forward_workers));
            forward_dif::<MODULUS, ROOT>(right, forward_workers);
        });
    }
}

fn pointwise_multiply<const MODULUS: u64>(left: &mut [u64], right: &[u64], workers: usize) {
    let workers = linear_worker_count(left.len(), workers);
    let chunk_len = left.len().div_ceil(workers);
    std::thread::scope(|scope| {
        let mut chunks = left.chunks_mut(chunk_len).zip(right.chunks(chunk_len));
        let (caller_left, caller_right) = chunks
            .next()
            .expect("a non-empty transform has a caller pointwise chunk");
        for (left, right) in chunks {
            let _ = scope.spawn(move || pointwise_chunk::<MODULUS>(left, right));
        }
        pointwise_chunk::<MODULUS>(caller_left, caller_right);
    });
}

fn pointwise_chunk<const MODULUS: u64>(left: &mut [u64], right: &[u64]) {
    for (lhs, &rhs) in left.iter_mut().zip(right) {
        *lhs = mul_mod::<MODULUS>(*lhs, rhs);
    }
}

fn linear_worker_count(len: usize, max_workers: usize) -> usize {
    debug_assert!(len != 0 && max_workers != 0);
    let maximum_by_work = (len / MIN_LINEAR_VALUES_PER_WORKER).max(1);
    let maximum = max_workers.min(maximum_by_work);
    1usize << maximum.ilog2()
}

fn copy_residues(values: &[u64], len: usize, workers: usize) -> Vec<u64> {
    let mut residues = vec![0u64; len];
    let workers = linear_worker_count(len, workers);
    let chunk_len = len.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut chunks = values[..len]
            .chunks(chunk_len)
            .zip(residues.chunks_mut(chunk_len));
        let (caller_values, caller_residues) = chunks
            .next()
            .expect("an NTT convolution has at least one residue");
        for (values, residues) in chunks {
            let _ = scope.spawn(move || residues.copy_from_slice(values));
        }
        caller_residues.copy_from_slice(caller_values);
    });
    residues
}

fn clear_pair(left: &mut [u64], right: &mut [u64], workers: usize) {
    debug_assert_eq!(left.len(), right.len());
    let workers = linear_worker_count(left.len().saturating_mul(2), workers);
    if workers == 1 {
        left.fill(0);
        right.fill(0);
        return;
    }
    let workers_per_buffer = workers / 2;
    std::thread::scope(|scope| {
        let _ = scope.spawn(|| parallel_fill_zero(left, workers_per_buffer));
        parallel_fill_zero(right, workers_per_buffer);
    });
}

fn parallel_fill_zero(values: &mut [u64], workers: usize) {
    let workers = linear_worker_count(values.len(), workers);
    if workers == 1 {
        values.fill(0);
        return;
    }
    let chunk_len = values.len().div_ceil(workers);
    std::thread::scope(|scope| {
        let mut chunks = values.chunks_mut(chunk_len);
        let caller_chunk = chunks
            .next()
            .expect("a non-empty transform has a caller clear chunk");
        for chunk in chunks {
            let _ = scope.spawn(move || chunk.fill(0));
        }
        caller_chunk.fill(0);
    });
}

fn square_mod<const MODULUS: u64, const ROOT: u64>(values: &mut [u64], max_contexts: usize) {
    forward_dif::<MODULUS, ROOT>(values, max_contexts);
    let workers = linear_worker_count(values.len(), max_contexts);
    let chunk_len = values.len().div_ceil(workers);
    std::thread::scope(|scope| {
        let mut chunks = values.chunks_mut(chunk_len);
        let caller_chunk = chunks
            .next()
            .expect("a non-empty transform has a caller square chunk");
        for chunk in chunks {
            let _ = scope.spawn(move || pointwise_square_chunk::<MODULUS>(chunk));
        }
        pointwise_square_chunk::<MODULUS>(caller_chunk);
    });
    transform_from_bit_reversed::<MODULUS, ROOT>(values, true, max_contexts);
}

fn pointwise_square_chunk<const MODULUS: u64>(values: &mut [u64]) {
    for value in values {
        *value = mul_mod::<MODULUS>(*value, *value);
    }
}

/// Transform input already in bit-reversed order into natural order.
fn transform_from_bit_reversed<const MODULUS: u64, const ROOT: u64>(
    values: &mut [u64],
    inverse: bool,
    max_contexts: usize,
) {
    debug_assert!(values.len().is_power_of_two());
    debug_assert!((MODULUS - 1).is_multiple_of(values.len() as u64));

    // After global bit reversal, every butterfly through `segment_len` stays
    // inside one aligned segment. Those early stages therefore run on
    // disjoint mutable slices with no locks or unsafe code. The remaining
    // log2(workers) stages join segments; those split their independent
    // blocks and butterfly lanes over the same worker budget below. A
    // power-of-two worker count is required to keep every segment aligned
    // to every radix-2 stage; rounding down also keeps this call within its
    // context budget. A radix-2 stage has only n/2 butterflies, so more
    // contexts cannot do work and would make the narrowest lane empty.
    let worker_limit = max_contexts.max(1).min((values.len() / 2).max(1));
    let workers = 1usize << worker_limit.ilog2();
    let segment_len = values.len() / workers;
    // Every segment is the same length, so one set of stage tables serves
    // them all.
    let tables = StageTables::new::<MODULUS, ROOT>(inverse, segment_len);
    if workers > 1 {
        std::thread::scope(|scope| {
            let mut segments = values.chunks_exact_mut(segment_len);
            let caller_segment = segments
                .next()
                .expect("a non-empty transform has a caller segment");
            for segment in segments {
                let tables = &tables;
                // Dropping the handle avoids a per-wave handle Vec; the scope
                // still joins every child and propagates an unjoined panic.
                let _ = scope
                    .spawn(move || dit_depth_first::<MODULUS>(segment, 2, segment_len, tables));
            }
            dit_depth_first::<MODULUS>(caller_segment, 2, segment_len, &tables);
        });
    } else {
        dit_depth_first::<MODULUS>(values, 2, segment_len, &tables);
    }
    transform_joining_stages::<MODULUS, ROOT>(
        values,
        inverse,
        segment_len.saturating_mul(2),
        workers,
    );

    if inverse {
        let inverse_len =
            Twiddle::new::<MODULUS>(pow_mod::<MODULUS>(values.len() as u64, MODULUS - 2));
        for value in values {
            *value = inverse_len.times::<MODULUS>(*value);
        }
    }
}

/// The forward transform by decimation in frequency: natural-order input to
/// a bit-reversed spectrum, in place.
///
/// Wide stages cross the segments the workers retain, so they run first,
/// each split over the workers; once the width reaches one segment the
/// remaining stages are independent and each worker takes its segment to
/// the end.
fn forward_dif<const MODULUS: u64, const ROOT: u64>(values: &mut [u64], max_contexts: usize) {
    debug_assert!(values.len().is_power_of_two());
    debug_assert!((MODULUS - 1).is_multiple_of(values.len() as u64));

    let worker_limit = max_contexts.max(1).min((values.len() / 2).max(1));
    let workers = 1usize << worker_limit.ilog2();
    let segment_len = values.len() / workers;

    let mut width = values.len();
    while width > segment_len {
        let root = stage_root::<MODULUS, ROOT>(width, false);
        dif_joining_stage::<MODULUS, ROOT>(values, width, root, workers);
        width >>= 1;
    }
    if segment_len < 2 {
        return;
    }
    let tables = StageTables::new::<MODULUS, ROOT>(false, segment_len);
    if workers > 1 {
        std::thread::scope(|scope| {
            let mut segments = values.chunks_exact_mut(segment_len);
            let caller_segment = segments
                .next()
                .expect("a non-empty transform has a caller segment");
            for segment in segments {
                let tables = &tables;
                let _ =
                    scope.spawn(move || dif_depth_first::<MODULUS>(segment, segment_len, tables));
            }
            dif_depth_first::<MODULUS>(caller_segment, segment_len, &tables);
        });
    } else {
        dif_depth_first::<MODULUS>(values, segment_len, &tables);
    }
}

/// The root of a stage of `width`: a primitive `width`-th root of unity, or
/// its inverse.
fn stage_root<const MODULUS: u64, const ROOT: u64>(width: usize, inverse: bool) -> u64 {
    let forward = pow_mod::<MODULUS>(ROOT, (MODULUS - 1) / width as u64);
    if inverse {
        pow_mod::<MODULUS>(forward, MODULUS - 2)
    } else {
        forward
    }
}

/// The forward stages of widths `width` down to 2 over `values`,
/// decimation in frequency, depth first: a wide span takes the stage that
/// splits it, then each half.
fn dif_depth_first<const MODULUS: u64>(values: &mut [u64], width: usize, tables: &StageTables) {
    let len = values.len();
    if len > CACHE_BLOCK_LEN {
        let (low, high) = values.split_at_mut(len / 2);
        if len <= width {
            let (root, table) = tables.stage(len);
            stage_pairs::<MODULUS, true>(low, high, root, table, 0);
        }
        dif_depth_first::<MODULUS>(low, width.min(low.len()), tables);
        dif_depth_first::<MODULUS>(high, width.min(high.len()), tables);
        return;
    }
    let mut width = width.min(values.len());
    while width >= 4 {
        let (root, table) = tables.stage(width);
        if width / 2 <= table.len() {
            stage_within_table::<MODULUS, true>(values, width, table);
        } else {
            for block in values.chunks_exact_mut(width) {
                let (low, high) = block.split_at_mut(width / 2);
                stage_pairs::<MODULUS, true>(low, high, root, table, 0);
            }
        }
        width >>= 1;
    }
    if width == 2 {
        stage_of_pairs::<MODULUS, true>(values);
    }
}

fn dif_joining_stage<const MODULUS: u64, const ROOT: u64>(
    values: &mut [u64],
    width: usize,
    root: u64,
    workers: usize,
) {
    let blocks = values.len() / width;
    debug_assert!(blocks <= workers);
    debug_assert_eq!(workers % blocks, 0);
    let contexts_per_block = workers / blocks;
    let table = stage_table::<MODULUS, ROOT>(false, width);
    std::thread::scope(|scope| {
        let mut block_slices = values.chunks_exact_mut(width);
        let caller_block = block_slices
            .next()
            .expect("a DIF joining stage has a caller block");
        for block in block_slices {
            let _ = scope.spawn(move || {
                stage_block::<MODULUS, true>(block, root, table, contexts_per_block)
            });
        }
        stage_block::<MODULUS, true>(caller_block, root, table, contexts_per_block);
    });
}

/// The stages of widths `width..=final_width` over `values`, decimation in
/// time, depth first: a span of [`CACHE_BLOCK_LEN`] values takes all its
/// stages while it is in cache, and a wider span its two halves and then
/// the stage that joins them.
fn dit_depth_first<const MODULUS: u64>(
    values: &mut [u64],
    width: usize,
    final_width: usize,
    tables: &StageTables,
) {
    let len = values.len();
    if len > CACHE_BLOCK_LEN && len > width {
        let (low, high) = values.split_at_mut(len / 2);
        dit_depth_first::<MODULUS>(low, width, final_width.min(low.len()), tables);
        dit_depth_first::<MODULUS>(high, width, final_width.min(high.len()), tables);
        if len <= final_width {
            let (root, table) = tables.stage(len);
            stage_pairs::<MODULUS, false>(low, high, root, table, 0);
        }
        return;
    }
    let final_width = final_width.min(values.len());
    let mut width = width;
    if width == 2 && final_width >= 2 {
        stage_of_pairs::<MODULUS, false>(values);
        width = 4;
    }
    while width <= final_width {
        let (root, table) = tables.stage(width);
        if width / 2 <= table.len() {
            stage_within_table::<MODULUS, false>(values, width, table);
        } else {
            for block in values.chunks_exact_mut(width) {
                let (low, high) = block.split_at_mut(width / 2);
                stage_pairs::<MODULUS, false>(low, high, root, table, 0);
            }
        }
        width <<= 1;
    }
}

/// The stage of width two: its one twiddle is one, so a butterfly is a sum
/// and a difference, the same in either decimation.
fn stage_of_pairs<const MODULUS: u64, const DIF: bool>(values: &mut [u64]) {
    for pair in values.chunks_exact_mut(2) {
        let (lhs, rhs) = (pair[0], pair[1]);
        let sum = lhs + rhs;
        pair[0] = if sum >= MODULUS { sum - MODULUS } else { sum };
        pair[1] = sub_mod::<MODULUS>(lhs, rhs);
    }
}

/// One stage whose table covers it, every block in one loop: a call a
/// stage, where a call a block cost a short stage more than its
/// butterflies.
fn stage_within_table<const MODULUS: u64, const DIF: bool>(
    values: &mut [u64],
    width: usize,
    table: &[Twiddle],
) {
    let table = &table[..width / 2];
    let one = Twiddle::new::<MODULUS>(1);
    for block in values.chunks_exact_mut(width) {
        let (low, high) = block.split_at_mut(width / 2);
        butterflies::<MODULUS, DIF, false>(low, high, table, one);
    }
}

/// Each stage's root and twiddle table for one transform, the tables the
/// process's kept ones: the depth-first order visits a stage's width many
/// times, and every worker's segment is the same length.
struct StageTables {
    /// Indexed by the stage's `log2(width)`.
    stages: Vec<(u64, &'static [Twiddle])>,
}

impl StageTables {
    fn new<const MODULUS: u64, const ROOT: u64>(inverse: bool, final_width: usize) -> Self {
        let stages = (0..=final_width.ilog2())
            .map(|bits| {
                let width = 1usize << bits;
                (
                    stage_root::<MODULUS, ROOT>(width, inverse),
                    stage_table::<MODULUS, ROOT>(inverse, width),
                )
            })
            .collect();
        Self { stages }
    }

    fn stage(&self, width: usize) -> (u64, &[Twiddle]) {
        let (root, table) = &self.stages[width.ilog2() as usize];
        (*root, table)
    }
}

/// The twiddle table of the stage of `width` under a prime, forward or
/// inverse, built on first use and kept for the process: the first
/// `min(width/2, 2^16)` powers of the stage's root.
fn stage_table<const MODULUS: u64, const ROOT: u64>(
    inverse: bool,
    width: usize,
) -> &'static [Twiddle] {
    use std::sync::OnceLock;
    // A cell a (prime, direction, log2 width): the widths a transform can
    // have, and the three primes by their index.
    const WIDTHS: usize = 56;
    static TABLES: [[[OnceLock<Vec<Twiddle>>; WIDTHS]; 2]; 3] =
        [const { [const { [const { OnceLock::new() }; WIDTHS] }; 2] }; 3];
    let prime = match MODULUS {
        PRIME_0 => 0,
        PRIME_1 => 1,
        PRIME_2 => 2,
        _ => unreachable!("the transform runs under its three primes alone"),
    };
    TABLES[prime][usize::from(inverse)][width.ilog2() as usize]
        .get_or_init(|| twiddles::<MODULUS>(stage_root::<MODULUS, ROOT>(width, inverse), width / 2))
}

/// Stages wider than a retained segment. There are fewer blocks than workers,
/// so blocks run concurrently and each block divides its independent
/// low/high butterfly pairs among the remaining contexts. Nested scoped
/// workers include their calling block worker in the budget: the total live
/// execution contexts is exactly `workers`, never `workers + callers`.
fn transform_joining_stages<const MODULUS: u64, const ROOT: u64>(
    values: &mut [u64],
    inverse: bool,
    mut width: usize,
    workers: usize,
) {
    while width <= values.len() {
        let root = stage_root::<MODULUS, ROOT>(width, inverse);
        let blocks = values.len() / width;
        debug_assert!(blocks < workers);
        debug_assert_eq!(workers % blocks, 0);
        let contexts_per_block = workers / blocks;
        let table = stage_table::<MODULUS, ROOT>(inverse, width);
        std::thread::scope(|scope| {
            let mut block_slices = values.chunks_exact_mut(width);
            let caller_block = block_slices
                .next()
                .expect("a joining stage has a caller block");
            for block in block_slices {
                let _ = scope.spawn(move || {
                    stage_block::<MODULUS, false>(block, root, table, contexts_per_block)
                });
            }
            stage_block::<MODULUS, false>(caller_block, root, table, contexts_per_block);
        });
        width <<= 1;
    }
}

/// One block of a stage, its pairs divided among `contexts` workers, each
/// chunk from its own position in the stage's twiddles.
fn stage_block<const MODULUS: u64, const DIF: bool>(
    block: &mut [u64],
    root: u64,
    table: &[Twiddle],
    contexts: usize,
) {
    let (low, high) = block.split_at_mut(block.len() / 2);
    debug_assert_eq!(low.len(), high.len());
    debug_assert_eq!(low.len() % contexts, 0);
    if contexts == 1 {
        stage_pairs::<MODULUS, DIF>(low, high, root, table, 0);
        return;
    }
    let chunk_len = low.len() / contexts;
    std::thread::scope(|scope| {
        let mut chunks = low
            .chunks_exact_mut(chunk_len)
            .zip(high.chunks_exact_mut(chunk_len));
        let (caller_low, caller_high) = chunks
            .next()
            .expect("a parallel butterfly has a caller chunk");
        for (chunk, (low, high)) in chunks.enumerate() {
            let start = (chunk + 1) * chunk_len;
            let _ = scope.spawn(move || stage_pairs::<MODULUS, DIF>(low, high, root, table, start));
        }
        stage_pairs::<MODULUS, DIF>(caller_low, caller_high, root, table, 0);
    });
}

/// A fixed multiplier under a modulus with its quotient precomputed, so
/// that a product by it reduces with one high multiplication and one
/// subtraction in place of a division: Shoup's form, as in V. Shoup's NTL
/// and Harvey (2014), §2.
#[derive(Clone, Copy)]
struct Twiddle {
    value: u64,
    /// `⌊value · 2^64 / MODULUS⌋`.
    quotient: u64,
}

impl Twiddle {
    fn new<const MODULUS: u64>(value: u64) -> Self {
        debug_assert!(value < MODULUS);
        Self {
            value,
            quotient: shoup_quotient::<MODULUS>(value),
        }
    }

    /// `lhs · value mod MODULUS` for any word `lhs`: the quotient estimate
    /// `⌊lhs · quotient / 2^64⌋` is the true quotient or one under it, so the
    /// remainder is below `2 · MODULUS`, which a word holds since the
    /// modulus is under `2^62`, and one subtraction settles it.
    #[inline]
    fn times<const MODULUS: u64>(self, lhs: u64) -> u64 {
        let estimate = ((u128::from(lhs) * u128::from(self.quotient)) >> 64) as u64;
        let remainder = lhs
            .wrapping_mul(self.value)
            .wrapping_sub(estimate.wrapping_mul(MODULUS));
        if remainder >= MODULUS {
            remainder - MODULUS
        } else {
            remainder
        }
    }
}

/// `⌊w · 2^64 / p⌋` for `w < p`, without a double-word division: with
/// `2^64 = p·⌊2^64/p⌋ + (2^64 mod p)`, it is `w·⌊2^64/p⌋ + ⌊w·(2^64 mod
/// p)/p⌋`, and the last quotient is Barrett's estimate on a product under
/// `2^124`, corrected.
fn shoup_quotient<const MODULUS: u64>(w: u64) -> u64 {
    let whole = u64::MAX / MODULUS;
    let rest = (u64::MAX % MODULUS + 1) % MODULUS;
    w * whole + div_wide::<MODULUS>(u128::from(w) * u128::from(rest)).0
}

/// `root^0, root^1, …` up to `count` entries or [`TWIDDLE_TABLE_LEN`],
/// whichever is fewer.
fn twiddles<const MODULUS: u64>(root: u64, count: usize) -> Vec<Twiddle> {
    let count = count.min(TWIDDLE_TABLE_LEN);
    let mut table = Vec::with_capacity(count);
    let mut weight = 1u64;
    for _ in 0..count {
        table.push(Twiddle::new::<MODULUS>(weight));
        weight = mul_mod::<MODULUS>(weight, root);
    }
    table
}

/// The butterflies of pairs `start..start + low.len()` of one stage, whose
/// twiddle at pair `j` is `root^j`: `table` holds the first powers, and a
/// pair past the table takes its entry under the power at its span's head.
/// Spans and chunks are powers of two, so a chunk is whole spans or lies
/// within one. Decimation in time applies the twiddle before the sum,
/// decimation in frequency (`DIF`) after the difference.
fn stage_pairs<const MODULUS: u64, const DIF: bool>(
    low: &mut [u64],
    high: &mut [u64],
    root: u64,
    table: &[Twiddle],
    start: usize,
) {
    debug_assert_eq!(low.len(), high.len());
    let span = table.len();
    if start == 0 && low.len() <= span {
        // The common call: a stage no wider than its table, from its head.
        butterflies::<MODULUS, DIF, false>(
            low,
            high,
            &table[..low.len()],
            Twiddle::new::<MODULUS>(1),
        );
        return;
    }
    let head = start - start % span;
    let mut scale = pow_mod::<MODULUS>(root, head as u64);
    let step = pow_mod::<MODULUS>(root, span as u64);
    let mut position = start;
    let mut offset = 0;
    while offset < low.len() {
        let within = position % span;
        let take = (span - within).min(low.len() - offset);
        let (low, high) = (
            &mut low[offset..offset + take],
            &mut high[offset..offset + take],
        );
        let table = &table[within..within + take];
        if scale == 1 {
            butterflies::<MODULUS, DIF, false>(low, high, table, Twiddle::new::<MODULUS>(1));
        } else {
            butterflies::<MODULUS, DIF, true>(low, high, table, Twiddle::new::<MODULUS>(scale));
        }
        offset += take;
        position += take;
        if position.is_multiple_of(span) {
            scale = mul_mod::<MODULUS>(scale, step);
        }
    }
}

/// One span of butterflies with twiddles `table[j]`, times `scale` when
/// `SCALED`: the scale is applied as a multiplication of its own, since
/// both it and the twiddle carry a quotient and their product would not.
/// Nothing carries from one pair to the next.
fn butterflies<const MODULUS: u64, const DIF: bool, const SCALED: bool>(
    low: &mut [u64],
    high: &mut [u64],
    table: &[Twiddle],
    scale: Twiddle,
) {
    for ((even, odd), &twiddle) in low.iter_mut().zip(high).zip(table) {
        let lhs = *even;
        if DIF {
            let rhs = *odd;
            let sum = lhs + rhs;
            *even = if sum >= MODULUS { sum - MODULUS } else { sum };
            let twisted = twiddle.times::<MODULUS>(sub_mod::<MODULUS>(lhs, rhs));
            *odd = if SCALED {
                scale.times::<MODULUS>(twisted)
            } else {
                twisted
            };
        } else {
            let scaled = if SCALED {
                scale.times::<MODULUS>(*odd)
            } else {
                *odd
            };
            let rhs = twiddle.times::<MODULUS>(scaled);
            let sum = lhs + rhs;
            *even = if sum >= MODULUS { sum - MODULUS } else { sum };
            *odd = sub_mod::<MODULUS>(lhs, rhs);
        }
    }
}

/// `lhs · rhs mod MODULUS` for residues `lhs, rhs < MODULUS`.
#[inline]
fn mul_mod<const MODULUS: u64>(lhs: u64, rhs: u64) -> u64 {
    reduce_wide::<MODULUS>(u128::from(lhs) * u128::from(rhs))
}

/// `x mod MODULUS` for `x < 2^124`, by Barrett's estimate of the quotient.
///
/// With `X = ⌊x/2^60⌋ < 2^64` and `M = ⌊2^124/p⌋ < 2^64` (`p > 2^60`), the
/// estimate `q̂ = ⌊X·M/2^64⌋` satisfies `q̂ ≤ X·2^60/p ≤ x/p`, and
/// `q̂ > X·2^60/p − X/2^64 − 1 > x/p − 2^60/p − 2 > x/p − 3`, so `q̂` is the
/// quotient or one or two under it, and `x − q̂·p < 3p < 2^64` is exact in
/// a word.
#[inline]
fn reduce_wide<const MODULUS: u64>(x: u128) -> u64 {
    debug_assert!(x < 1u128 << 124);
    let m = ((1u128 << 124) / u128::from(MODULUS)) as u64;
    let estimate = ((u128::from((x >> 60) as u64) * u128::from(m)) >> 64) as u64;
    let mut remainder = (x as u64).wrapping_sub(estimate.wrapping_mul(MODULUS));
    if remainder >= MODULUS {
        remainder -= MODULUS;
    }
    if remainder >= MODULUS {
        remainder -= MODULUS;
    }
    remainder
}

/// `(⌊x/p⌋, x mod p)` for `x < 2^124`: Barrett's estimate and its two
/// corrections, counted.
fn div_wide<const MODULUS: u64>(x: u128) -> (u64, u64) {
    debug_assert!(x < 1u128 << 124);
    let m = ((1u128 << 124) / u128::from(MODULUS)) as u64;
    let mut quotient = ((u128::from((x >> 60) as u64) * u128::from(m)) >> 64) as u64;
    let mut remainder = (x as u64).wrapping_sub(quotient.wrapping_mul(MODULUS));
    while remainder >= MODULUS {
        remainder -= MODULUS;
        quotient += 1;
    }
    (quotient, remainder)
}

fn pow_mod<const MODULUS: u64>(mut base: u64, mut exponent: u64) -> u64 {
    let mut result = 1u64;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = mul_mod::<MODULUS>(result, base);
        }
        base = mul_mod::<MODULUS>(base, base);
        exponent >>= 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64 (Steele, Lea & Flood, OOPSLA 2014).
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn is_prime(n: u64) -> bool {
        // Deterministic Miller–Rabin for 64-bit integers: these bases suffice
        // (Jaeschke 1993; Sorenson & Webster 2015).
        if n < 2 {
            return false;
        }
        for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
            if n.is_multiple_of(p) {
                return n == p;
            }
        }
        let (mut d, mut s) = (n - 1, 0);
        while d % 2 == 0 {
            d /= 2;
            s += 1;
        }
        let pow = |mut b: u128, mut e: u64| {
            let mut r = 1u128;
            while e > 0 {
                if e & 1 == 1 {
                    r = r * b % u128::from(n);
                }
                b = b * b % u128::from(n);
                e >>= 1;
            }
            r as u64
        };
        'bases: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
            let mut x = pow(u128::from(a), d);
            if x == 1 || x == n - 1 {
                continue;
            }
            for _ in 1..s {
                x = ((u128::from(x) * u128::from(x)) % u128::from(n)) as u64;
                if x == n - 1 {
                    continue 'bases;
                }
            }
            return false;
        }
        true
    }

    /// The primes are `c·2^k + 1` with `k ≥ 55`, between `2^60` and `2^62`,
    /// prime, and each root is primitive: `root^((p−1)/q) ≠ 1` for every
    /// prime `q` dividing `p − 1`, which are 2 and the factors of `c`.
    #[test]
    fn primes_and_roots_hold() {
        for (p, root, c, k, c_factors) in [
            (PRIME_0, ROOT_0, 29u64, 57u32, &[29u64][..]),
            (PRIME_1, ROOT_1, 69, 55, &[3, 23]),
            (PRIME_2, ROOT_2, 57, 55, &[3, 19]),
        ] {
            assert_eq!(p, c * (1u64 << k) + 1);
            assert!(k >= 55 && (1u64 << 60) < p && p < (1u64 << 62));
            assert!(is_prime(p), "{p}");
            for q in c_factors.iter().chain(&[2]) {
                assert_ne!(
                    pow_mod_test(p, root, (p - 1) / q),
                    1,
                    "{p}: {root} is not primitive"
                );
            }
            // The least primitive root: every smaller base fails some factor.
            for g in 2..root {
                assert!(
                    c_factors
                        .iter()
                        .chain(&[2])
                        .any(|q| pow_mod_test(p, g, (p - 1) / q) == 1),
                    "{p}: {g} is a smaller primitive root"
                );
            }
        }
    }

    fn pow_mod_test(p: u64, base: u64, mut e: u64) -> u64 {
        let (mut r, mut b) = (1u128, u128::from(base));
        while e > 0 {
            if e & 1 == 1 {
                r = r * b % u128::from(p);
            }
            b = b * b % u128::from(p);
            e >>= 1;
        }
        r as u64
    }

    /// A transform of `2^55` values keeps every coefficient exact and no
    /// longer one would: `n·(2^64 − 1)²` against `p₀·p₁·p₂`, compared through
    /// their bit lengths and then exactly.
    #[test]
    fn transform_len_is_exact() {
        // p₀·p₁·p₂ as (high, low) 128-bit halves of a 256-bit product.
        let p0p1 = P0P1;
        let (lo, hi) = mul_256(p0p1, u128::from(PRIME_2));
        // n·(2^64 − 1)² for n = 2^55: (2^64 − 1)² < 2^128, times 2^55.
        let square = u128::from(u64::MAX) * u128::from(u64::MAX);
        let (bound_lo, bound_hi) = (square << 55, square >> (128 - 55));
        assert!(
            (bound_hi, bound_lo) < (hi, lo),
            "2^55 coefficients are exact"
        );
        let (bound_lo, bound_hi) = (square << 56, square >> (128 - 56));
        assert!(
            (bound_hi, bound_lo) > (hi, lo),
            "2^56 coefficients would not be"
        );
        assert_eq!(
            MAX_TRANSFORM_LEN as u64,
            if usize::BITS > 55 {
                1u64 << 55
            } else {
                1u64 << (usize::BITS - 1)
            }
        );
        assert!((PRIME_1 - 1).is_multiple_of(1 << 55) && (PRIME_2 - 1).is_multiple_of(1 << 55));
    }

    fn mul_256(a: u128, b: u128) -> (u128, u128) {
        let (a0, a1) = (a as u64 as u128, a >> 64);
        let (b0, b1) = (b as u64 as u128, b >> 64);
        let ll = a0 * b0;
        let lh = a0 * b1;
        let hl = a1 * b0;
        let hh = a1 * b1;
        let mid = (ll >> 64) + (lh as u64 as u128) + (hl as u64 as u128);
        let lo = (ll as u64 as u128) | (mid << 64);
        let hi = hh + (lh >> 64) + (hl >> 64) + (mid >> 64);
        (lo, hi)
    }

    /// A transform of 2^18 values on one worker, whose segment is the whole
    /// and whose wide stages take their table under a span's leading power:
    /// `(B^n − 1)·r = r·B^n − r`, a product every worker count leaves
    /// through the same code but these stages.
    #[test]
    fn wide_stages_on_one_worker_multiply_exactly() {
        let mut state = 0x3c6e_f372_fe94_f82bu64;
        let n = 150_000usize;
        let r = BigUint::from_limbs((0..140_000).map(|_| next(&mut state) | 1).collect());
        let all_ones = BigUint::from_limbs(vec![u64::MAX; n]);
        let mut shifted = r.clone();
        shifted.shl_bits(64 * n);
        let expected = shifted.sub(&r);
        assert_eq!(multiply_with_workers(&all_ones, &r, 1), expected);
        assert_eq!(multiply_with_workers(&r, &all_ones, 4), expected);
        // (B^n − 1)² = B^{2n} − 2·B^n + 1, on one worker.
        let mut twice = BigUint::from_u64(2);
        twice.shl_bits(64 * n);
        let mut top = BigUint::from_u64(1);
        top.shl_bits(128 * n);
        let expected_square = top.sub(&twice).add(&BigUint::from_u64(1));
        assert_eq!(
            crate::parallel::with_budget(1, || square(&all_ones)),
            expected_square
        );
    }

    #[test]
    fn garner_constants_hold() {
        assert_eq!(
            mul_mod::<PRIME_1>(reduce_word::<PRIME_1>(PRIME_0), INVERSE_P0_MOD_P1),
            1
        );
        assert_eq!(
            mul_mod::<PRIME_2>(reduce_wide::<PRIME_2>(P0P1), INVERSE_P0P1_MOD_P2),
            1
        );
    }

    /// The reductions against `u128` arithmetic: Barrett on products of
    /// residues and on the widest inputs it admits, the Shoup form against
    /// the plain product, and the quotient by multiplication against the
    /// division it replaces.
    #[test]
    fn reductions_match_division() {
        fn check<const P: u64>(state: &mut u64) {
            for i in 0..20_000 {
                let (a, b) = (next(state) % P, next(state) % P);
                assert_eq!(
                    mul_mod::<P>(a, b),
                    (u128::from(a) * u128::from(b) % u128::from(P)) as u64
                );
                let x = if i % 2 == 0 {
                    u128::from(next(state)) << 60 | u128::from(next(state))
                } else {
                    u128::from(a) * u128::from(b)
                } & ((1u128 << 124) - 1);
                assert_eq!(reduce_wide::<P>(x), (x % u128::from(P)) as u64);
                let (q, r) = div_wide::<P>(x);
                assert_eq!(
                    (u128::from(q), u128::from(r)),
                    (x / u128::from(P), x % u128::from(P))
                );
                let w = Twiddle::new::<P>(a);
                assert_eq!(w.quotient, ((u128::from(a) << 64) / u128::from(P)) as u64);
                let any = next(state);
                assert_eq!(
                    w.times::<P>(any),
                    (u128::from(any) * u128::from(a) % u128::from(P)) as u64
                );
            }
            for x in [
                0u128,
                1,
                (1u128 << 124) - 1,
                u128::from(P) - 1,
                u128::from(P),
                3 * u128::from(P) - 1,
            ] {
                assert_eq!(reduce_wide::<P>(x), (x % u128::from(P)) as u64);
            }
        }
        let mut state = 0x5bd1_e995_7f4a_7c15u64;
        check::<PRIME_0>(&mut state);
        check::<PRIME_1>(&mut state);
        check::<PRIME_2>(&mut state);
    }

    /// Garner's reconstruction against values below `p₀·p₁·p₂` reduced the
    /// long way: the extremes, and values drawn across the whole range,
    /// the top limb set.
    #[test]
    fn crt_recovers_coefficients() {
        fn residue(limbs: [u64; 3], p: u64) -> u64 {
            let p = u128::from(p);
            let top = u128::from(limbs[2]) % p;
            let mid = ((top << 64) | u128::from(limbs[1])) % p;
            (((mid << 64) | u128::from(limbs[0])) % p) as u64
        }
        let recovers = |value: [u64; 3]| {
            let got = crt_three(
                residue(value, PRIME_0),
                residue(value, PRIME_1),
                residue(value, PRIME_2),
            );
            assert_eq!(got, value);
        };
        // p₀·p₁·p₂ as (high, low) halves of a 256-bit product: 184 bits.
        let (bound_lo, bound_hi) = mul_256(P0P1, u128::from(PRIME_2));
        assert_eq!(bound_hi.leading_zeros(), 128 - (184 - 128));
        let largest = if bound_lo == 0 {
            [u64::MAX, u64::MAX, (bound_hi - 1) as u64]
        } else {
            [
                (bound_lo - 1) as u64,
                ((bound_lo - 1) >> 64) as u64,
                bound_hi as u64,
            ]
        };
        for value in [
            [0, 0, 0],
            [1, 0, 0],
            [u64::MAX, 0, 0],
            [u64::MAX, u64::MAX, 0],
            largest,
        ] {
            recovers(value);
        }
        let mut state = 0x6a09_e667_f3bc_c909u64;
        let mut drawn = 0;
        while drawn < 20_000 {
            let value = [next(&mut state), next(&mut state), next(&mut state) >> 8];
            let low = (u128::from(value[1]) << 64) | u128::from(value[0]);
            if (u128::from(value[2]), low) >= (bound_hi, bound_lo) {
                continue;
            }
            recovers(value);
            drawn += 1;
        }
    }
}
