//! Volatile zeroization, the crate's sole non-test `unsafe` exception
//! (the other audited site is the test probe that reads a scrubbed
//! buffer's raw tail back; the crate root names both).
//!
//! Verbatim from the parent crate's audited helper (`cryptography-rs`,
//! `src/ct.rs`), carried here so [`crate::BigUint`] can wipe its limbs on
//! drop without a dependency.
//!
//! The one primitive below overwrites memory that the caller still owns and
//! can address at the moment of the call. That is its entire scope: it is a
//! hygiene measure against values lingering in freed heap blocks, not a
//! side-channel countermeasure, and it does not make any operation in this
//! crate constant-time. The crate-root scope statement governs.

use core::ptr;
use core::sync::atomic::{compiler_fence, Ordering};

/// Overwrites every element of `slice` with its `Default` value.
///
/// Why volatile rather than `slice.fill(T::default())`: a store to memory
/// that is about to be dropped or returned to the allocator is dead by the
/// compiler's own reasoning, and the optimizer is entitled to delete it.
/// `ptr::write_volatile` is specified as an externally observable effect, so
/// the store must be emitted and may not be merged away; the trailing
/// `compiler_fence(SeqCst)` bars the compiler from sinking those stores past
/// the deallocation or buffer reuse that follows the call.
///
/// What the call guarantees: the addresses `slice` currently spans hold
/// `T::default()` when it returns.
///
/// What it does not guarantee, and what a consumer must not read into it:
///
/// - **It does not follow a reallocation.** A `Vec` that grows copies its
///   elements into a new block and releases the old one; the abandoned block
///   still holds the original bytes, and no later scrub of the live buffer
///   reaches them. Only the buffer passed in is touched.
/// - **It does not touch spare capacity.** `slice` is the live elements; the
///   tail between length and capacity, and any limbs stranded when a buffer
///   is truncated without this helper being called on the abandoned range,
///   keep their contents.
/// - **It is not a side-channel control.** The loop is an ordinary
///   data-independent walk, but it neither hides nor equalizes the timing of
///   anything else, and it cannot recall copies already made elsewhere:
///   register and stack spills, cache lines, pages written to swap, or a core
///   dump taken before the call.
///
/// # Safety argument for the `unsafe` block
///
/// The pointer is derived from `iter_mut()`, so for each element it is
/// non-null, aligned, and valid for a write of `T` for the duration of the
/// borrow. `write_volatile` does not run a destructor on the overwritten
/// value; `T: Copy` means there is none to run.
#[allow(unsafe_code)] // the non-test audited exception to the crate-wide deny: volatile scrub
pub fn zeroize_slice<T: Copy + Default>(slice: &mut [T]) {
    for item in slice.iter_mut() {
        unsafe { ptr::write_volatile(std::ptr::from_mut::<T>(item), T::default()) };
    }
    compiler_fence(Ordering::SeqCst);
}
