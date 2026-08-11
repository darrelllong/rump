//! Volatile zeroization, the crate's sole `unsafe` exception.
//!
//! Verbatim from the parent crate's audited helper (`cryptography-rs`,
//! `src/ct.rs`), carried here so [`crate::BigUint`] can wipe its limbs on
//! drop without a dependency.

use core::ptr;
use core::sync::atomic::{compiler_fence, Ordering};

/// Overwrites every element of `slice` with its `Default` value.
///
/// Uses `ptr::write_volatile` so the compiler cannot prove the writes are dead
/// and elide them (which it would be allowed to do for ordinary assignments to
/// memory that is about to go out of scope).  The `compiler_fence` prevents
/// reordering the volatile stores with subsequent deallocation or reuse of the
/// backing memory.
#[allow(unsafe_code)] // sole audited exception to the crate-wide deny: volatile scrub
pub fn zeroize_slice<T: Copy + Default>(slice: &mut [T]) {
    for item in slice.iter_mut() {
        unsafe { ptr::write_volatile(std::ptr::from_mut::<T>(item), T::default()) };
    }
    compiler_fence(Ordering::SeqCst);
}
