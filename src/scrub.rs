//! Optional volatile scrubbing of the caller's live slice. This is a no-op
//! without `wipe`. It does not erase copies or make arithmetic constant-time.

#[cfg(feature = "wipe")]
use core::ptr;
#[cfg(feature = "wipe")]
use core::sync::atomic::{compiler_fence, Ordering};

/// Overwrite each element with `T::default()` using volatile stores.
/// Volatile writes prevent dead-store elimination of this overwrite; the
/// compiler fence orders it before subsequent buffer reuse.
///
/// Only the supplied slice is covered. Spare capacity, buffers abandoned by
/// reallocation, prior copies, registers, swap, and earlier dumps are not.
/// This is not a side-channel control.
///
/// # Safety argument for the `unsafe` block
///
/// The pointer is derived from `iter_mut()`, so for each element it is
/// non-null, aligned, and valid for a write of `T` for the duration of the
/// borrow. `write_volatile` does not run a destructor on the overwritten
/// value; `T: Copy` means there is none to run.
#[cfg(feature = "wipe")]
#[allow(unsafe_code)] // the one non-test exception to the crate-wide deny: volatile scrub
pub(crate) fn zeroize_slice<T: Copy + Default>(slice: &mut [T]) {
    for item in slice.iter_mut() {
        unsafe { ptr::write_volatile(std::ptr::from_mut::<T>(item), T::default()) };
    }
    compiler_fence(Ordering::SeqCst);
}

/// No-op in builds without `wipe`; callers need no feature conditionals.
#[cfg(not(feature = "wipe"))]
#[inline(always)]
pub(crate) fn zeroize_slice<T: Copy + Default>(_slice: &mut [T]) {}
