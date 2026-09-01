//! Exact large-integer multiplication by number-theoretic transform.
//!
//! Limbs are split into base-2^16 digits and convolved modulo two NTT
//! primes.  Their product exceeds every possible raw convolution coefficient
//! at the supported transform lengths, so two-prime CRT recovers each integer
//! coefficient uniquely; an ordinary carry pass then returns to base 2^64.
//! This is the modular-transform form of Schönhage and Strassen's fast integer
//! multiplication (Computing 7 (1971), 281–292), with iterative radix-2
//! Cooley–Tukey transforms. Digit expansion supplies bit-reversed forward
//! input directly. Scoped workers divide disjoint stages and butterfly lanes
//! without exceeding reported machine parallelism; squaring uses one transform
//! buffer and pointwise self-products.

use super::BigUint;

const DIGIT_BITS: usize = 16;
const DIGITS_PER_LIMB: usize = 64 / DIGIT_BITS;
const DIGIT_MASK: u64 = (1 << DIGIT_BITS) - 1;

// Both primes are c·2^k + 1 and admit the listed primitive root.  The second
// prime sets the common transform ceiling at 2^26.
const PRIME_0: u64 = 2_013_265_921; // 15·2^27 + 1
const ROOT_0: u64 = 31;
const PRIME_1: u64 = 1_811_939_329; // 27·2^26 + 1
const ROOT_1: u64 = 13;
const MAX_TRANSFORM_LEN: usize = 1 << 26;
const PRIME_PRODUCT: u64 = PRIME_0 * PRIME_1;

// Linear transform passes need enough values per context to repay one scoped
// worker wave. This grain is measured by the ignored phase/scaling probes and
// limits workers by work size in addition to the caller's hardware ceiling.
const MIN_LINEAR_VALUES_PER_WORKER: usize = 1 << 16;

// Below 2^16 coefficients NTT loses to the recursive multiplication ladder
// even with four contexts on the crossover host. Keeping forced small-kernel
// tests serial also prevents thread-launch time from dominating their work.
const PARALLEL_TRANSFORM_MIN_LEN: usize = 1 << 16;

/// Padded transform length for full-width operands, when supported.
pub(super) fn transform_len(lhs_limbs: usize, rhs_limbs: usize) -> Option<usize> {
    let lhs_digits = lhs_limbs.checked_mul(DIGITS_PER_LIMB)?;
    let rhs_digits = rhs_limbs.checked_mul(DIGITS_PER_LIMB)?;
    let convolution_len = lhs_digits
        .checked_add(rhs_digits)
        .and_then(|sum| sum.checked_sub(1))?;
    convolution_len
        .checked_next_power_of_two()
        .filter(|&len| len <= MAX_TRANSFORM_LEN)
}

/// Number of execution contexts the automatic transform will actually use.
pub(super) fn automatic_worker_count(transform_len: usize) -> usize {
    let available = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    worker_count(transform_len, available)
}

/// Select a radix-compatible worker count from a hard context ceiling.
///
/// With `w = 2^k` fixed segments, `log2(n) - k` stages run in one retained
/// scoped wave and the `k` stages that join segments each require another
/// synchronized wave. Charging one unit of launch/barrier latency per joining
/// wave gives `(log2(n) - k) / w + k`; choose its exact integer-rational
/// minimum instead of imposing an unrelated maximum. At the present 2^26
/// transform ceiling the model itself never selects more than 16 workers.
pub(super) fn worker_count(transform_len: usize, max_contexts: usize) -> usize {
    debug_assert!(transform_len.is_power_of_two());
    if transform_len < PARALLEL_TRANSFORM_MIN_LEN || max_contexts <= 1 {
        return 1;
    }
    let depth = transform_len.ilog2() as usize;
    let maximum = max_contexts.min(transform_len);
    let maximum_power = 1usize << maximum.ilog2();
    let mut best_workers = 1usize;
    let mut best_numerator = depth;
    let mut workers = 2usize;
    let mut joining_stages = 1usize;
    while workers <= maximum_power {
        let numerator = depth - joining_stages + joining_stages * workers;
        if numerator * best_workers < best_numerator * workers {
            best_workers = workers;
            best_numerator = numerator;
        }
        workers <<= 1;
        joining_stages += 1;
    }
    best_workers
}

/// Multiply two non-zero values through an exact two-prime NTT convolution.
pub(super) fn multiply(lhs: &BigUint, rhs: &BigUint) -> BigUint {
    let contexts = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    multiply_impl(lhs, rhs, contexts)
}

/// Square one non-zero value through one transform buffer and exact CRT.
pub(super) fn square(value: &BigUint) -> BigUint {
    debug_assert!(!value.limbs.is_empty());

    let digit_len = significant_digit_len(value);
    let convolution_len = digit_len
        .checked_mul(2)
        .and_then(|sum| sum.checked_sub(1))
        .expect("NTT square convolution length fits usize");
    let transform_len = convolution_len
        .checked_next_power_of_two()
        .expect("NTT square transform length fits usize");
    assert!(
        transform_len <= MAX_TRANSFORM_LEN,
        "NTT square transform exceeds the supported 2^26 coefficients"
    );
    let coefficient_bound = (digit_len as u128) * u128::from(DIGIT_MASK) * u128::from(DIGIT_MASK);
    assert!(coefficient_bound < u128::from(PRIME_PRODUCT));
    let workers = automatic_worker_count(transform_len);

    let mut values = vec![0u64; transform_len];
    write_digits_bit_reversed(value, digit_len, &mut values, workers);
    square_mod::<PRIME_0, ROOT_0>(&mut values, workers);
    let residues_0 = copy_residues(&values, convolution_len, workers);

    values.fill(0);
    write_digits_bit_reversed(value, digit_len, &mut values, workers);
    square_mod::<PRIME_1, ROOT_1>(&mut values, workers);
    reconstruct(&residues_0, &mut values, convolution_len, workers)
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

    let lhs_digits = significant_digit_len(lhs);
    let rhs_digits = significant_digit_len(rhs);
    let convolution_len = lhs_digits
        .checked_add(rhs_digits)
        .and_then(|sum| sum.checked_sub(1))
        .expect("NTT convolution length fits usize");
    let transform_len = convolution_len
        .checked_next_power_of_two()
        .expect("NTT transform length fits usize");
    assert!(
        transform_len <= MAX_TRANSFORM_LEN,
        "NTT transform exceeds the supported 2^26 coefficients"
    );

    // The maximum coefficient is overlap·(2^16-1)^2.  At the largest
    // supported transform, overlap <= 2^25, while PRIME_0·PRIME_1 > 2^61.
    // Keeping this executable assertion beside the CRT prevents a later base
    // or transform-limit change from silently invalidating exact recovery.
    let coefficient_bound =
        (lhs_digits.min(rhs_digits) as u128) * u128::from(DIGIT_MASK) * u128::from(DIGIT_MASK);
    assert!(coefficient_bound < u128::from(PRIME_PRODUCT));
    let workers = select_workers(transform_len);
    debug_assert!(workers.is_power_of_two() && workers <= transform_len);

    let mut left = vec![0u64; transform_len];
    let mut right = vec![0u64; transform_len];
    write_input_pair_bit_reversed(
        lhs, lhs_digits, rhs, rhs_digits, &mut left, &mut right, workers,
    );
    convolve_mod::<PRIME_0, ROOT_0>(&mut left, &mut right, workers);
    let residues_0 = copy_residues(&right, convolution_len, workers);

    clear_pair(&mut left, &mut right, workers);
    write_input_pair_bit_reversed(
        lhs, lhs_digits, rhs, rhs_digits, &mut left, &mut right, workers,
    );
    convolve_mod::<PRIME_1, ROOT_1>(&mut left, &mut right, workers);

    reconstruct(&residues_0, &mut right, convolution_len, workers)
}

/// Wall-clock phase breakdown for the ignored NTT profiling probe.
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

/// Multiply while measuring the high-level phases without changing their
/// implementation. This exists only in test builds so production calls do not
/// pay for timestamps or carry a profiling interface.
#[cfg(test)]
pub(super) fn multiply_profiled(
    lhs: &BigUint,
    rhs: &BigUint,
    workers: usize,
) -> (BigUint, MultiplicationProfile) {
    use std::time::Instant;

    assert!(
        workers.is_power_of_two(),
        "NTT worker count must be a power of two"
    );
    let lhs_digits = significant_digit_len(lhs);
    let rhs_digits = significant_digit_len(rhs);
    let convolution_len = lhs_digits + rhs_digits - 1;
    let transform_len = convolution_len.next_power_of_two();
    assert!(transform_len <= MAX_TRANSFORM_LEN);
    let workers = workers.min(transform_len);

    let started = Instant::now();
    let mut left = vec![0u64; transform_len];
    let mut right = vec![0u64; transform_len];
    let allocate = started.elapsed();

    let started = Instant::now();
    write_input_pair_bit_reversed(
        lhs, lhs_digits, rhs, rhs_digits, &mut left, &mut right, workers,
    );
    let mut prepare_inputs = started.elapsed();

    let started = Instant::now();
    forward_transform_pair::<PRIME_0, ROOT_0>(&mut left, &mut right, workers);
    let mut forward = started.elapsed();
    let started = Instant::now();
    pointwise_multiply::<PRIME_0>(&mut left, &right, workers);
    let mut pointwise = started.elapsed();
    let started = Instant::now();
    inverse_to_natural::<PRIME_0, ROOT_0>(&mut left, &mut right, workers);
    let mut inverse = started.elapsed();

    let started = Instant::now();
    let residues_0 = copy_residues(&right, convolution_len, workers);
    let residue_copy = started.elapsed();

    let started = Instant::now();
    clear_pair(&mut left, &mut right, workers);
    let clear = started.elapsed();
    let started = Instant::now();
    write_input_pair_bit_reversed(
        lhs, lhs_digits, rhs, rhs_digits, &mut left, &mut right, workers,
    );
    prepare_inputs += started.elapsed();

    let started = Instant::now();
    forward_transform_pair::<PRIME_1, ROOT_1>(&mut left, &mut right, workers);
    forward += started.elapsed();
    let started = Instant::now();
    pointwise_multiply::<PRIME_1>(&mut left, &right, workers);
    pointwise += started.elapsed();
    let started = Instant::now();
    inverse_to_natural::<PRIME_1, ROOT_1>(&mut left, &mut right, workers);
    inverse += started.elapsed();

    let started = Instant::now();
    let product = reconstruct(&residues_0, &mut right, convolution_len, workers);
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

/// Reconstruct coefficients and carry directly into packed 64-bit limbs.
fn reconstruct(
    residues_0: &[u32],
    residues_1: &mut [u64],
    convolution_len: usize,
    workers: usize,
) -> BigUint {
    debug_assert_eq!(residues_0.len(), convolution_len);
    debug_assert!(residues_1.len() >= convolution_len);
    let coefficients = &mut residues_1[..convolution_len];
    reconstruct_coefficients(residues_0, coefficients, workers);
    let mut limbs = Vec::with_capacity((convolution_len + DIGITS_PER_LIMB) / 4);
    let mut word = 0u64;
    let mut digit_in_word = 0usize;
    let mut carry = 0u128;
    let mut push_digit = |digit: u64| {
        word |= digit << (DIGIT_BITS * digit_in_word);
        digit_in_word += 1;
        if digit_in_word == DIGITS_PER_LIMB {
            limbs.push(word);
            word = 0;
            digit_in_word = 0;
        }
    };

    for &raw_coefficient in coefficients.iter() {
        let coefficient = u128::from(raw_coefficient) + carry;
        push_digit((coefficient as u64) & DIGIT_MASK);
        carry = coefficient >> DIGIT_BITS;
    }
    while carry != 0 {
        push_digit((carry as u64) & DIGIT_MASK);
        carry >>= DIGIT_BITS;
    }
    if digit_in_word != 0 {
        limbs.push(word);
    }
    BigUint::from_limbs(limbs)
}

fn reconstruct_coefficients(residues_0: &[u32], coefficients: &mut [u64], workers: usize) {
    let workers = linear_worker_count(coefficients.len(), workers);
    let chunk_len = coefficients.len().div_ceil(workers);
    std::thread::scope(|scope| {
        let mut chunks = residues_0
            .chunks(chunk_len)
            .zip(coefficients.chunks_mut(chunk_len));
        let (caller_0, caller_coefficients) = chunks
            .next()
            .expect("an NTT convolution has at least one coefficient");
        for (residues_0, coefficients) in chunks {
            let _ = scope.spawn(move || reconstruct_coefficient_chunk(residues_0, coefficients));
        }
        reconstruct_coefficient_chunk(caller_0, caller_coefficients);
    });
}

fn reconstruct_coefficient_chunk(residues_0: &[u32], coefficients: &mut [u64]) {
    for (&residue_0, residue_1) in residues_0.iter().zip(coefficients) {
        *residue_1 = crt_two(u64::from(residue_0), *residue_1);
    }
}

fn significant_digit_len(value: &BigUint) -> usize {
    let top = *value
        .limbs
        .last()
        .expect("NTT multiplication receives non-zero operands");
    (value.limbs.len() - 1)
        .checked_mul(DIGITS_PER_LIMB)
        .and_then(|digits| {
            digits.checked_add((64 - top.leading_zeros() as usize).div_ceil(DIGIT_BITS))
        })
        .expect("NTT digit length fits usize")
}

/// Expand directly into the permutation expected by a decimation-in-time
/// transform. This removes four full-array bit-reversal passes per product;
/// zero padding is already present in the destination allocation/fill.
fn write_digits_bit_reversed(
    value: &BigUint,
    digit_len: usize,
    digits: &mut [u64],
    workers: usize,
) {
    debug_assert!(digits.len().is_power_of_two());
    debug_assert!(workers.is_power_of_two() && workers <= digits.len());
    if workers == 1 {
        write_digit_segment_bit_reversed(value, digit_len, digits, 0, 1);
        return;
    }

    let segment_len = digits.len() / workers;
    std::thread::scope(|scope| {
        let mut segments = digits.chunks_exact_mut(segment_len).enumerate();
        let (caller_index, caller_segment) = segments
            .next()
            .expect("a non-empty transform has a caller input segment");
        for (segment_index, segment) in segments {
            let _ = scope.spawn(move || {
                write_digit_segment_bit_reversed(value, digit_len, segment, segment_index, workers)
            });
        }
        write_digit_segment_bit_reversed(value, digit_len, caller_segment, caller_index, workers);
    });
}

/// Fill one aligned segment of a globally bit-reversed transform input.
///
/// Reversing an index swaps the high bits that select the destination segment
/// with the low bits of the source index. Consequently segment `s` owns every
/// source digit whose low `log2(workers)` bits equal `reverse(s)`. This gives
/// each worker a disjoint mutable slice without locks or unsafe code.
fn write_digit_segment_bit_reversed(
    value: &BigUint,
    digit_len: usize,
    segment: &mut [u64],
    segment_index: usize,
    workers: usize,
) {
    let worker_bits = workers.ilog2();
    let source_offset = if worker_bits == 0 {
        0
    } else {
        segment_index.reverse_bits() >> (usize::BITS - worker_bits)
    };
    let segment_bits = segment.len().ilog2();
    for source_index in (source_offset..digit_len).step_by(workers) {
        let local_source = source_index >> worker_bits;
        let destination = if segment_bits == 0 {
            0
        } else {
            local_source.reverse_bits() >> (usize::BITS - segment_bits)
        };
        let limb = value.limbs[source_index / DIGITS_PER_LIMB];
        let shift = DIGIT_BITS * (source_index % DIGITS_PER_LIMB);
        segment[destination] = (limb >> shift) & DIGIT_MASK;
    }
}

fn write_input_pair_bit_reversed(
    lhs: &BigUint,
    lhs_digits: usize,
    rhs: &BigUint,
    rhs_digits: usize,
    left: &mut [u64],
    right: &mut [u64],
    workers: usize,
) {
    debug_assert_eq!(left.len(), right.len());
    if workers == 1 {
        write_digits_bit_reversed(lhs, lhs_digits, left, 1);
        write_digits_bit_reversed(rhs, rhs_digits, right, 1);
        return;
    }

    let input_workers = workers / 2;
    std::thread::scope(|scope| {
        let _ = scope.spawn(|| write_digits_bit_reversed(lhs, lhs_digits, left, input_workers));
        write_digits_bit_reversed(rhs, rhs_digits, right, input_workers);
    });
}

/// Recover `x < PRIME_0·PRIME_1` from its two residues.
fn crt_two(residue_0: u64, residue_1: u64) -> u64 {
    // PRIME_0^-1 mod PRIME_1 = -9.  Thus
    // x = residue_0 + PRIME_0·((residue_1-residue_0)·(-9) mod PRIME_1).
    let residue_0_mod_1 = if residue_0 >= PRIME_1 {
        residue_0 - PRIME_1
    } else {
        residue_0
    };
    let delta = if residue_1 >= residue_0_mod_1 {
        residue_1 - residue_0_mod_1
    } else {
        residue_1 + PRIME_1 - residue_0_mod_1
    };
    let negated = (9 * delta) % PRIME_1;
    let multiplier = if negated == 0 { 0 } else { PRIME_1 - negated };
    residue_0 + PRIME_0 * multiplier
}

fn convolve_mod<const MODULUS: u64, const ROOT: u64>(
    left: &mut [u64],
    right: &mut [u64],
    workers: usize,
) {
    debug_assert_eq!(left.len(), right.len());
    forward_transform_pair::<MODULUS, ROOT>(left, right, workers);
    pointwise_multiply::<MODULUS>(left, right, workers);
    // `right` is dead after the pointwise product, so it becomes the
    // natural-order inverse output without allocating another transform-sized
    // buffer.
    inverse_to_natural::<MODULUS, ROOT>(left, right, workers);
}

fn forward_transform_pair<const MODULUS: u64, const ROOT: u64>(
    left: &mut [u64],
    right: &mut [u64],
    workers: usize,
) {
    if workers == 1 {
        transform_from_bit_reversed::<MODULUS, ROOT>(left, false, 1);
        transform_from_bit_reversed::<MODULUS, ROOT>(right, false, 1);
    } else {
        // The two forward transforms are independent. Running both at once
        // with half the worker budget removes one whole transform from the
        // critical path while keeping the total live contexts at `workers`.
        // The inverse has no independent peer and uses the full budget.
        let forward_workers = workers / 2;
        std::thread::scope(|scope| {
            let _ = scope.spawn(|| {
                transform_from_bit_reversed::<MODULUS, ROOT>(left, false, forward_workers)
            });
            transform_from_bit_reversed::<MODULUS, ROOT>(right, false, forward_workers);
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

fn copy_residues(values: &[u64], len: usize, workers: usize) -> Vec<u32> {
    let mut residues = vec![0u32; len];
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
            let _ = scope.spawn(move || copy_residue_chunk(values, residues));
        }
        copy_residue_chunk(caller_values, caller_residues);
    });
    residues
}

fn copy_residue_chunk(values: &[u64], residues: &mut [u32]) {
    for (&value, residue) in values.iter().zip(residues) {
        *residue = value as u32;
    }
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
    transform_from_bit_reversed::<MODULUS, ROOT>(values, false, max_contexts);
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
    transform::<MODULUS, ROOT>(values, true, max_contexts);
}

fn pointwise_square_chunk<const MODULUS: u64>(values: &mut [u64]) {
    for value in values {
        *value = mul_mod::<MODULUS>(*value, *value);
    }
}

/// In-place iterative radix-2 Cooley–Tukey transform.
fn transform<const MODULUS: u64, const ROOT: u64>(
    values: &mut [u64],
    inverse: bool,
    max_contexts: usize,
) {
    debug_assert!(values.len().is_power_of_two());
    debug_assert!((MODULUS - 1).is_multiple_of(values.len() as u64));

    // Bit-reversal permutation, incrementally updating the reversed index.
    let mut reversed = 0usize;
    for index in 1..values.len() {
        let mut bit = values.len() >> 1;
        while reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if index < reversed {
            values.swap(index, reversed);
        }
    }

    transform_from_bit_reversed::<MODULUS, ROOT>(values, inverse, max_contexts);
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
    // log2(workers) stages join segments; those split their independent blocks
    // and butterfly lanes over the same worker budget below. A power-of-two
    // worker count is required to keep every segment aligned to every radix-2
    // stage; rounding down also guarantees that this call never exceeds its
    // detected or explicitly supplied context budget.
    // A radix-2 stage has only n/2 butterflies, so more contexts cannot do
    // work and would make the narrowest per-context lane empty.
    let worker_limit = max_contexts.max(1).min((values.len() / 2).max(1));
    let workers = 1usize << worker_limit.ilog2();
    let segment_len = values.len() / workers;
    if workers > 1 {
        std::thread::scope(|scope| {
            let mut segments = values.chunks_exact_mut(segment_len);
            let caller_segment = segments
                .next()
                .expect("a non-empty transform has a caller segment");
            for segment in segments {
                // Dropping the handle avoids a per-wave handle Vec; the scope
                // still joins every child and propagates an unjoined panic.
                let _ = scope.spawn(move || {
                    transform_stages::<MODULUS, ROOT>(segment, inverse, 2, segment_len)
                });
            }
            transform_stages::<MODULUS, ROOT>(caller_segment, inverse, 2, segment_len);
        });
    } else {
        transform_stages::<MODULUS, ROOT>(values, inverse, 2, segment_len);
    }
    transform_joining_stages::<MODULUS, ROOT>(
        values,
        inverse,
        segment_len.saturating_mul(2),
        workers,
    );

    if inverse {
        let inverse_len = pow_mod::<MODULUS>(values.len() as u64, MODULUS - 2);
        for value in values {
            *value = mul_mod::<MODULUS>(*value, inverse_len);
        }
    }
}

/// Inverse a natural-order spectrum into natural-order coefficients.
///
/// A decimation-in-frequency inverse emits bit-reversed coefficients without
/// a leading permutation. The final parallel gather both restores natural
/// order and applies `n^-1`, replacing the serial bit-reversal and serial
/// normalization passes used by the general in-place transform.
fn inverse_to_natural<const MODULUS: u64, const ROOT: u64>(
    values: &mut [u64],
    output: &mut [u64],
    max_contexts: usize,
) {
    debug_assert_eq!(values.len(), output.len());
    debug_assert!(values.len().is_power_of_two());
    debug_assert!((MODULUS - 1).is_multiple_of(values.len() as u64));

    let worker_limit = max_contexts.max(1).min((values.len() / 2).max(1));
    let workers = 1usize << worker_limit.ilog2();
    let segment_len = values.len() / workers;

    // Wide DIF stages cross retained-segment boundaries, so execute them
    // first. Once the width reaches one segment, all remaining stages are
    // independent and each worker retains its slice to the end.
    let mut width = values.len();
    while width > segment_len {
        let root = inverse_stage_root::<MODULUS, ROOT>(width);
        inverse_dif_joining_stage::<MODULUS>(values, width, root, workers);
        width >>= 1;
    }
    if workers > 1 {
        std::thread::scope(|scope| {
            let mut segments = values.chunks_exact_mut(segment_len);
            let caller_segment = segments
                .next()
                .expect("a non-empty inverse transform has a caller segment");
            for segment in segments {
                let _ =
                    scope.spawn(move || inverse_dif_stages::<MODULUS, ROOT>(segment, segment_len));
            }
            inverse_dif_stages::<MODULUS, ROOT>(caller_segment, segment_len);
        });
    } else {
        inverse_dif_stages::<MODULUS, ROOT>(values, segment_len);
    }

    let inverse_len = pow_mod::<MODULUS>(values.len() as u64, MODULUS - 2);
    gather_bit_reversed_normalized::<MODULUS>(values, output, inverse_len, workers);
}

fn inverse_stage_root<const MODULUS: u64, const ROOT: u64>(width: usize) -> u64 {
    let forward = pow_mod::<MODULUS>(ROOT, (MODULUS - 1) / width as u64);
    pow_mod::<MODULUS>(forward, MODULUS - 2)
}

fn inverse_dif_stages<const MODULUS: u64, const ROOT: u64>(values: &mut [u64], mut width: usize) {
    while width >= 2 {
        let root = inverse_stage_root::<MODULUS, ROOT>(width);
        for block in values.chunks_exact_mut(width) {
            let (low, high) = block.split_at_mut(width / 2);
            inverse_dif_pairs::<MODULUS>(low, high, root, 1);
        }
        width >>= 1;
    }
}

fn inverse_dif_joining_stage<const MODULUS: u64>(
    values: &mut [u64],
    width: usize,
    root: u64,
    workers: usize,
) {
    let blocks = values.len() / width;
    debug_assert!(blocks <= workers);
    debug_assert_eq!(workers % blocks, 0);
    let contexts_per_block = workers / blocks;
    std::thread::scope(|scope| {
        let mut block_slices = values.chunks_exact_mut(width);
        let caller_block = block_slices
            .next()
            .expect("a DIF joining stage has a caller block");
        for block in block_slices {
            let _ =
                scope.spawn(move || inverse_dif_block::<MODULUS>(block, root, contexts_per_block));
        }
        inverse_dif_block::<MODULUS>(caller_block, root, contexts_per_block);
    });
}

fn inverse_dif_block<const MODULUS: u64>(block: &mut [u64], root: u64, contexts: usize) {
    let (low, high) = block.split_at_mut(block.len() / 2);
    debug_assert_eq!(low.len(), high.len());
    debug_assert_eq!(low.len() % contexts, 0);
    if contexts == 1 {
        inverse_dif_pairs::<MODULUS>(low, high, root, 1);
        return;
    }

    let chunk_len = low.len() / contexts;
    std::thread::scope(|scope| {
        let mut chunks = low
            .chunks_exact_mut(chunk_len)
            .zip(high.chunks_exact_mut(chunk_len));
        let (caller_low, caller_high) = chunks
            .next()
            .expect("a parallel DIF butterfly has a caller chunk");
        for (chunk, (low, high)) in chunks.enumerate() {
            let start = (chunk + 1) * chunk_len;
            let _ = scope.spawn(move || {
                inverse_dif_pairs::<MODULUS>(
                    low,
                    high,
                    root,
                    pow_mod::<MODULUS>(root, start as u64),
                )
            });
        }
        inverse_dif_pairs::<MODULUS>(caller_low, caller_high, root, 1);
    });
}

fn inverse_dif_pairs<const MODULUS: u64>(
    low: &mut [u64],
    high: &mut [u64],
    root: u64,
    mut weight: u64,
) {
    debug_assert_eq!(low.len(), high.len());
    for (even, odd) in low.iter_mut().zip(high) {
        let lhs = *even;
        let rhs = *odd;
        let sum = lhs + rhs;
        *even = if sum >= MODULUS { sum - MODULUS } else { sum };
        let difference = if lhs >= rhs {
            lhs - rhs
        } else {
            lhs + MODULUS - rhs
        };
        *odd = mul_mod::<MODULUS>(difference, weight);
        weight = mul_mod::<MODULUS>(weight, root);
    }
}

fn gather_bit_reversed_normalized<const MODULUS: u64>(
    source: &[u64],
    output: &mut [u64],
    inverse_len: u64,
    workers: usize,
) {
    let transform_bits = source.len().ilog2();
    let chunk_len = output.len() / workers;
    std::thread::scope(|scope| {
        let mut chunks = output.chunks_exact_mut(chunk_len).enumerate();
        let (caller_index, caller_chunk) = chunks
            .next()
            .expect("a non-empty inverse output has a caller chunk");
        for (chunk_index, chunk) in chunks {
            let start = chunk_index * chunk_len;
            let _ = scope.spawn(move || {
                gather_bit_reversed_chunk::<MODULUS>(
                    source,
                    chunk,
                    start,
                    transform_bits,
                    inverse_len,
                )
            });
        }
        gather_bit_reversed_chunk::<MODULUS>(
            source,
            caller_chunk,
            caller_index * chunk_len,
            transform_bits,
            inverse_len,
        );
    });
}

fn gather_bit_reversed_chunk<const MODULUS: u64>(
    source: &[u64],
    output: &mut [u64],
    start: usize,
    transform_bits: u32,
    inverse_len: u64,
) {
    for (offset, value) in output.iter_mut().enumerate() {
        let index = start + offset;
        let reversed = if transform_bits == 0 {
            0
        } else {
            index.reverse_bits() >> (usize::BITS - transform_bits)
        };
        *value = mul_mod::<MODULUS>(source[reversed], inverse_len);
    }
}

fn transform_stages<const MODULUS: u64, const ROOT: u64>(
    values: &mut [u64],
    inverse: bool,
    mut width: usize,
    final_width: usize,
) {
    while width <= final_width {
        let mut root = pow_mod::<MODULUS>(ROOT, (MODULUS - 1) / width as u64);
        if inverse {
            root = pow_mod::<MODULUS>(root, MODULUS - 2);
        }
        for block in values.chunks_exact_mut(width) {
            let (low, high) = block.split_at_mut(width / 2);
            butterfly_pairs::<MODULUS>(low, high, root, 1);
        }
        width <<= 1;
    }
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
        let mut root = pow_mod::<MODULUS>(ROOT, (MODULUS - 1) / width as u64);
        if inverse {
            root = pow_mod::<MODULUS>(root, MODULUS - 2);
        }
        let blocks = values.len() / width;
        debug_assert!(blocks < workers);
        debug_assert_eq!(workers % blocks, 0);
        let contexts_per_block = workers / blocks;
        std::thread::scope(|scope| {
            let mut block_slices = values.chunks_exact_mut(width);
            let caller_block = block_slices
                .next()
                .expect("a joining stage has a caller block");
            for block in block_slices {
                let _ = scope
                    .spawn(move || butterfly_block::<MODULUS>(block, root, contexts_per_block));
            }
            butterfly_block::<MODULUS>(caller_block, root, contexts_per_block);
        });
        width <<= 1;
    }
}

fn butterfly_block<const MODULUS: u64>(block: &mut [u64], root: u64, contexts: usize) {
    let (low, high) = block.split_at_mut(block.len() / 2);
    debug_assert_eq!(low.len(), high.len());
    debug_assert_eq!(low.len() % contexts, 0);
    if contexts == 1 {
        butterfly_pairs::<MODULUS>(low, high, root, 1);
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
            let _ = scope.spawn(move || {
                butterfly_pairs::<MODULUS>(low, high, root, pow_mod::<MODULUS>(root, start as u64))
            });
        }
        butterfly_pairs::<MODULUS>(caller_low, caller_high, root, 1);
    });
}

fn butterfly_pairs<const MODULUS: u64>(
    low: &mut [u64],
    high: &mut [u64],
    root: u64,
    mut weight: u64,
) {
    debug_assert_eq!(low.len(), high.len());
    for (even, odd) in low.iter_mut().zip(high) {
        let lhs = *even;
        let rhs = mul_mod::<MODULUS>(*odd, weight);
        let sum = lhs + rhs;
        *even = if sum >= MODULUS { sum - MODULUS } else { sum };
        *odd = if lhs >= rhs {
            lhs - rhs
        } else {
            lhs + MODULUS - rhs
        };
        weight = mul_mod::<MODULUS>(weight, root);
    }
}

#[inline]
fn mul_mod<const MODULUS: u64>(lhs: u64, rhs: u64) -> u64 {
    // Both moduli are below 2^31, so the product is below 2^62 and cannot
    // overflow.  Keeping MODULUS const lets the optimizer replace division by
    // each fixed prime with multiplication by a reciprocal.
    (lhs * rhs) % MODULUS
}

fn pow_mod<const MODULUS: u64>(mut base: u64, mut exponent: u64) -> u64 {
    let mut result = 1u64;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = mul_mod::<MODULUS>(result, base);
        }
        exponent >>= 1;
        if exponent != 0 {
            base = mul_mod::<MODULUS>(base, base);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roots_generate_the_required_power_of_two_subgroups() {
        let is_prime_by_trial_division = |candidate: u64| {
            if candidate < 2 || candidate.is_multiple_of(2) {
                return candidate == 2;
            }
            let mut divisor = 3u64;
            while divisor * divisor <= candidate {
                if candidate.is_multiple_of(divisor) {
                    return false;
                }
                divisor += 2;
            }
            true
        };
        assert!(is_prime_by_trial_division(PRIME_0));
        assert!(is_prime_by_trial_division(PRIME_1));
        assert_eq!(PRIME_0 - 1, 15 * (1 << 27));
        assert_eq!(PRIME_1 - 1, 27 * (1 << 26));

        // The distinct prime factors of PRIME_0-1 are 2, 3, 5; those of
        // PRIME_1-1 are 2, 3. Non-unity at each quotient proves full order.
        for factor in [2, 3, 5] {
            assert_ne!(pow_mod::<PRIME_0>(ROOT_0, (PRIME_0 - 1) / factor), 1);
        }
        for factor in [2, 3] {
            assert_ne!(pow_mod::<PRIME_1>(ROOT_1, (PRIME_1 - 1) / factor), 1);
        }
    }

    #[test]
    fn crt_round_trips_boundary_values() {
        for value in [
            0,
            1,
            PRIME_1 - 1,
            PRIME_1,
            PRIME_0,
            PRIME_PRODUCT - 2,
            PRIME_PRODUCT - 1,
        ] {
            assert_eq!(crt_two(value % PRIME_0, value % PRIME_1), value);
        }
    }

    #[test]
    fn transform_round_trip() {
        for len in [1usize, 2, 4, 8, 32, 256] {
            let original: Vec<u64> = (0..len)
                .map(|index| (index as u64 * 1_234_567 + 89) % PRIME_0)
                .collect();
            let mut transformed = original.clone();
            transform::<PRIME_0, ROOT_0>(&mut transformed, false, 1);
            transform::<PRIME_0, ROOT_0>(&mut transformed, true, 1);
            assert_eq!(transformed, original);
        }
    }

    #[test]
    fn dif_inverse_matches_original_at_every_context_limit() {
        for len in [1usize, 2, 4, 8, 32, 256, 4096] {
            let original: Vec<u64> = (0..len)
                .map(|index| (index as u64 * 7_654_321 + 123) % PRIME_0)
                .collect();
            let mut spectrum = original.clone();
            transform::<PRIME_0, ROOT_0>(&mut spectrum, false, 1);
            for contexts in [1usize, 2, 3, 4, 5, 8] {
                let mut transformed = spectrum.clone();
                let mut actual = vec![0u64; len];
                inverse_to_natural::<PRIME_0, ROOT_0>(&mut transformed, &mut actual, contexts);
                assert_eq!(actual, original, "DIF inverse at {contexts} contexts");
            }
        }
    }

    #[test]
    fn parallel_bit_reversed_input_matches_serial_for_partial_limbs() {
        let values = [
            BigUint::from_limbs(vec![1]),
            BigUint::from_limbs(vec![u64::MAX]),
            BigUint::from_limbs(vec![0x0123_4567_89ab_cdef, 1]),
            BigUint::from_limbs(vec![u64::MAX, 0, 0x1234_5678]),
            BigUint::from_limbs(vec![0, u64::MAX, 7, 0, 1]),
        ];
        for value in values {
            let digit_len = significant_digit_len(&value);
            let transform_len = (digit_len * 2).next_power_of_two();
            let mut expected = vec![0u64; transform_len];
            write_digits_bit_reversed(&value, digit_len, &mut expected, 1);
            for workers in [2usize, 4, 8].into_iter().filter(|&w| w <= transform_len) {
                let mut actual = vec![0u64; transform_len];
                write_digits_bit_reversed(&value, digit_len, &mut actual, workers);
                assert_eq!(actual, expected, "bit-reversed input at {workers} workers");
            }
        }
    }

    #[test]
    fn transform_is_independent_of_context_limit() {
        let original: Vec<u64> = (0..4096)
            .map(|index| (index as u64 * 1_234_567 + 89) % PRIME_0)
            .collect();
        let mut serial = original.clone();
        transform::<PRIME_0, ROOT_0>(&mut serial, false, 1);
        for contexts in [2usize, 3, 4, 5, 8] {
            let mut parallel = original.clone();
            transform::<PRIME_0, ROOT_0>(&mut parallel, false, contexts);
            assert_eq!(parallel, serial, "forward transform at {contexts} contexts");
            transform::<PRIME_0, ROOT_0>(&mut parallel, true, contexts);
            assert_eq!(
                parallel, original,
                "inverse transform at {contexts} contexts"
            );
        }
    }

    #[test]
    fn transform_length_ceiling_is_exact() {
        // Four base-2^16 digits per limb and two equal operands require just
        // under eight transform coefficients per limb.
        assert_eq!(transform_len(1 << 23, 1 << 23), Some(MAX_TRANSFORM_LEN));
        assert_eq!(transform_len((1 << 23) + 1, (1 << 23) + 1), None);
    }

    #[test]
    fn worker_count_is_hardware_bounded_and_geometry_selected() {
        assert_eq!(worker_count(1 << 15, 256), 1);
        assert_eq!(worker_count(1 << 16, 1), 1);
        assert_eq!(worker_count(1 << 16, 2), 2);
        assert_eq!(worker_count(1 << 16, 6), 4);
        assert_eq!(worker_count(1 << 16, 256), 8);
        assert_eq!(worker_count(1 << 19, 256), 16);
        assert_eq!(worker_count(MAX_TRANSFORM_LEN, 256), 16);
        for available in 1usize..=256 {
            let workers = worker_count(MAX_TRANSFORM_LEN, available);
            assert!(workers <= available);
            assert!(workers.is_power_of_two());
        }
    }
}
