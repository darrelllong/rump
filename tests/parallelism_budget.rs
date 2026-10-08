//! The thread budget through the public surface: what a consumer that fans
//! out on threads of its own can rely on, on a machine of any size.
//!
//! The crate's own fan-outs are tested inside it (`src/parallel.rs`); here
//! is the contract a caller sees: `budget` is never zero and never more than
//! the machine reports, `with_budget` holds on the calling thread alone and
//! is put back, and the arithmetic that fans out gives the same answer at
//! every budget.

use rump::number_theory::{primes_below, product_tree, remainder_tree, CrtBasis};
use rump::parallelism::{budget, with_budget};
use rump::BigUint;

/// What the machine reports, as `budget` reads it outside any `with_budget`.
fn machine() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

/// A fixed, cheap generator: the tests want operands wide enough to fan
/// out over, not random ones.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// An odd value of `limbs` limbs with its top bit set.
fn operand(seed: u64, limbs: usize) -> BigUint {
    let mut state = seed;
    let mut words: Vec<u64> = (0..limbs).map(|_| splitmix(&mut state)).collect();
    words[0] |= 1;
    words[limbs - 1] |= 1 << 63;
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    BigUint::from_le_bytes(&bytes)
}

/// `2^p - 1` for every prime `p` in `low..high`: pairwise coprime, since
/// `gcd(2^a - 1, 2^b - 1) = 2^gcd(a, b) - 1`, so a CRT basis accepts them,
/// and together wide enough that the trees over them fan out.
fn mersenne_moduli(low: u64, high: u64) -> Vec<BigUint> {
    primes_below(high)
        .into_iter()
        .filter(|&p| p >= low)
        .map(|p| {
            let mut power = BigUint::one();
            power.shl_bits(usize::try_from(p).expect("a small exponent"));
            power.sub(&BigUint::one())
        })
        .collect()
}

#[test]
fn the_budget_is_never_zero_and_never_more_than_the_machine() {
    let machine = machine();
    assert!(machine >= 1);
    assert_eq!(budget(), machine);
    for asked in [0usize, 1, 2, 3, machine, machine + 1, usize::MAX] {
        let inside = with_budget(asked, budget);
        assert_eq!(inside, asked.clamp(1, machine), "with_budget({asked})");
        assert_eq!(budget(), machine, "put back after with_budget({asked})");
    }
}

#[test]
fn a_budget_is_the_calling_threads_and_the_others_keep_theirs() {
    // A consumer that fans out on threads of its own gives each a share by
    // setting it on that thread; what it set on its own thread is not seen
    // on theirs, and what they set is not seen on its.
    let machine = machine();
    with_budget(1, || {
        let seen: Vec<(usize, usize)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (1..=4)
                .map(|share| {
                    scope.spawn(move || {
                        let before = budget();
                        let inside = with_budget(share, budget);
                        assert_eq!(budget(), before);
                        (before, inside)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("the thread does not panic"))
                .collect()
        });
        for (share, (before, inside)) in (1..=4).zip(seen) {
            assert_eq!(before, machine, "a new thread starts at the machine's");
            assert_eq!(inside, share.min(machine));
        }
        assert_eq!(budget(), 1, "the caller's own budget is untouched");
    });
    assert_eq!(budget(), machine);
}

#[test]
fn a_panic_inside_with_budget_puts_the_budget_back() {
    let machine = machine();
    let caught = std::panic::catch_unwind(|| {
        with_budget(1, || {
            assert_eq!(budget(), 1);
            panic!("the work fails");
        })
    });
    assert!(caught.is_err());
    assert_eq!(budget(), machine);
}

#[test]
fn the_trees_and_the_crt_are_the_same_at_every_budget() {
    // Enough work that the trees fan out on a machine of several cores, and
    // the same answer whether they may use one thread, a few, or all.
    const SEED: u64 = 0x0b0d_6e7d_0000_0001;
    let values = mersenne_moduli(300, 2000);
    assert!(values.len() > 200, "{} moduli", values.len());
    let modulus = operand(SEED, 24);
    let serial = with_budget(1, || {
        let tree = product_tree(&values);
        let remainders = remainder_tree(&tree, &modulus);
        (tree.root().cloned(), remainders)
    });
    for (value, remainder) in values.iter().zip(&serial.1).step_by(31) {
        assert_eq!(*remainder, modulus.rem(value), "batched vs direct");
    }
    let residues: Vec<BigUint> = serial.1.clone();
    let combined = with_budget(1, || {
        let basis = CrtBasis::new(&values).expect("pairwise coprime moduli");
        basis.combine(&residues)
    });
    assert_eq!(
        combined,
        modulus.rem(serial.0.as_ref().expect("a root")),
        "the CRT recovers the modulus reduced by the product"
    );
    for asked in [2usize, 3, 5, machine(), usize::MAX] {
        let fanned = with_budget(asked, || {
            let tree = product_tree(&values);
            let remainders = remainder_tree(&tree, &modulus);
            let basis = CrtBasis::new(&values).expect("pairwise coprime moduli");
            (tree.root().cloned(), remainders, basis.combine(&residues))
        });
        assert_eq!(fanned.0, serial.0, "root at budget {asked}");
        assert_eq!(fanned.1, serial.1, "remainders at budget {asked}");
        assert_eq!(fanned.2, combined, "crt at budget {asked}");
    }
}
