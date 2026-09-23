//! A caller's field arithmetic on `BigUint` can run without touching the
//! allocator: a Mersenne fold multiply built only from the into-storage and
//! in-place operations makes no allocation once its buffers have their
//! capacity. Counted through the global allocator, so a call that quietly
//! allocates fails here rather than showing up as a profile.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rump::BigUint;

struct Counting;

static CALLS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        CALLS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Multiplication modulo the Mersenne prime `2^k − 1` by folding: the bits
/// above `k` weigh `2^k ≡ 1`, so the high part is added back to the low
/// part, twice, and one conditional subtraction finishes. Every step writes
/// into storage the caller keeps.
struct MersenneFold {
    k: usize,
    modulus: BigUint,
    product: BigUint,
    high: BigUint,
}

impl MersenneFold {
    fn new(k: usize) -> Self {
        let mut modulus = BigUint::one();
        modulus.shl_bits(k);
        modulus -= &BigUint::one();
        Self {
            k,
            modulus,
            product: BigUint::zero(),
            high: BigUint::zero(),
        }
    }

    /// `out ← a·b mod 2^k − 1`, for `a, b < 2^k − 1`.
    fn mul(&mut self, out: &mut BigUint, a: &BigUint, b: &BigUint) {
        self.product.mul_into(a, b);
        for _ in 0..2 {
            self.high.clone_from(&self.product);
            self.high.shr_bits(self.k);
            self.product.keep_low_bits(self.k);
            self.product += &self.high;
        }
        if self.product >= self.modulus {
            self.product -= &self.modulus;
        }
        out.clone_from(&self.product);
    }
}

fn draw(state: &mut u64, k: usize) -> BigUint {
    let mut limbs = Vec::new();
    for _ in 0..k.div_ceil(64) {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        limbs.push(*state);
    }
    let mut value = BigUint::zero();
    for limb in limbs.iter().rev() {
        value.shl_bits(64);
        value += &BigUint::from_u64(*limb);
    }
    value.keep_low_bits(k);
    value
}

#[test]
fn a_mersenne_fold_multiply_makes_no_allocation() {
    // The prime the secret-sharing field uses, and a wider one so the fold
    // crosses limb boundaries.
    for k in [127usize, 521] {
        let mut field = MersenneFold::new(k);
        let mut state = 0x666f_6c64_0000_0000u64 | k as u64;
        let a = draw(&mut state, k);
        let b = draw(&mut state, k);
        let mut out = BigUint::zero();
        // The first call sizes every buffer.
        field.mul(&mut out, &a, &b);

        // The oracle: plain product and remainder, allocating freely, taken
        // before the counted region.
        let mut expected = a.clone();
        expected = expected.mul(&b).rem(&field.modulus);
        let before = CALLS.load(Ordering::Relaxed);
        for _ in 0..1_000 {
            field.mul(&mut out, &a, &b);
        }
        let after = CALLS.load(Ordering::Relaxed);
        assert_eq!(out, expected, "fold modulo 2^{k} - 1");
        assert_eq!(
            after - before,
            0,
            "allocations in 1000 folds modulo 2^{k} - 1"
        );
    }
}
