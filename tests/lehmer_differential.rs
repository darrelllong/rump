//! Differential test of the Lehmer-based `gcd`, `gcd_extended`, and
//! `mod_inverse` against the classical Euclid implementations they replaced.
//!
//! The naive versions are reproduced verbatim here as oracles: they share no
//! code with the Lehmer engine, so agreement across a wide operand sweep plus
//! structured corner cases certifies the fast path computes the exact same
//! quotient sequence (hence the exact same gcd and Bézout cofactors).

use rump::{gcd, gcd_extended, jacobi, mod_inverse, BigInt, BigUint};

// ── the oracles: classical Euclid, exactly as rump shipped before Lehmer ──

fn gcd_naive(lhs: &BigUint, rhs: &BigUint) -> BigUint {
    let mut current = lhs.clone();
    let mut next = rhs.clone();
    while !next.is_zero() {
        let remainder = current.modulo(&next);
        current = next;
        next = remainder;
    }
    current
}

fn gcd_extended_naive(a: &BigUint, b: &BigUint) -> (BigUint, BigInt, BigInt) {
    let mut old_r = a.clone();
    let mut r = b.clone();
    let mut old_s = BigInt::from_biguint(BigUint::one());
    let mut s = BigInt::zero();
    let mut old_t = BigInt::zero();
    let mut t = BigInt::from_biguint(BigUint::one());
    while !r.is_zero() {
        let (quotient, remainder) = old_r.div_rem(&r);
        old_r = r;
        r = remainder;
        let next_s = old_s.sub_ref(&s.mul_biguint_ref(&quotient));
        old_s = s;
        s = next_s;
        let next_t = old_t.sub_ref(&t.mul_biguint_ref(&quotient));
        old_t = t;
        t = next_t;
    }
    (old_r, old_s, old_t)
}

/// The division-per-step Jacobi symbol, exactly as rump shipped before the
/// division-free rewrite — the oracle for `jacobi`.
fn jacobi_naive(a: &BigUint, n: &BigUint) -> Option<i8> {
    if n.is_zero() || !n.is_odd() {
        return None;
    }
    let mut a = a.modulo(n);
    let mut n = n.clone();
    let mut sign = 1i8;
    while !a.is_zero() {
        let mut twos = 0usize;
        while !a.bit(twos) {
            twos += 1;
        }
        if twos % 2 == 1 {
            let n_mod_8 = n.rem_u64(8);
            if n_mod_8 == 3 || n_mod_8 == 5 {
                sign = -sign;
            }
        }
        a.shr_bits(twos);
        if a.rem_u64(4) == 3 && n.rem_u64(4) == 3 {
            sign = -sign;
        }
        core::mem::swap(&mut a, &mut n);
        a = a.modulo(&n);
    }
    if n.is_one() {
        Some(sign)
    } else {
        Some(0)
    }
}

fn mod_inverse_naive(a: &BigUint, n: &BigUint) -> Option<BigUint> {
    if n.is_zero() {
        return None;
    }
    let mut t = BigInt::zero();
    let mut next_t = BigInt::from_biguint(BigUint::one());
    let mut r = n.clone();
    let mut next_r = a.modulo(n);
    while !next_r.is_zero() {
        let (quotient, remainder) = r.div_rem(&next_r);
        let coefficient = t.sub_ref(&next_t.mul_biguint_ref(&quotient));
        t = next_t;
        next_t = coefficient;
        r = next_r;
        next_r = remainder;
    }
    if !r.is_one() {
        return None;
    }
    Some(t.modulo_positive(n))
}

// ── a small deterministic RNG so failures reproduce ──

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A random non-negative integer of up to `bits` bits, decoded from bytes
    /// so it exercises the full limb range (not a fixed limb count).
    fn draw(&mut self, bits: usize) -> BigUint {
        let bytes = bits.div_ceil(8).max(1);
        let mut buf = vec![0u8; bytes];
        let mut i = 0;
        while i < buf.len() {
            let word = self.next_u64().to_le_bytes();
            let take = word.len().min(buf.len() - i);
            buf[i..i + take].copy_from_slice(&word[..take]);
            i += take;
        }
        // Trim to exactly `bits` by masking the top byte.
        let excess = bytes * 8 - bits;
        if excess > 0 {
            let last = buf.len() - 1;
            buf[last] &= 0xFFu8 >> excess;
        }
        BigUint::from_be_bytes(&{
            buf.reverse();
            buf
        })
    }
}

fn bezout_holds(a: &BigUint, b: &BigUint, g: &BigUint, s: &BigInt, t: &BigInt) -> bool {
    let lhs = s.mul_biguint_ref(a).add_ref(&t.mul_biguint_ref(b));
    lhs == BigInt::from_biguint(g.clone())
}

#[test]
fn gcd_family_matches_classical_euclid_over_random_sweep() {
    let mut rng = SplitMix64 {
        state: 0xA1B2_C3D4_E5F6_0718,
    };
    // Mix of sizes: sub-limb, around limb boundaries, and multi-limb up past
    // the point where Lehmer batching dominates.
    let sizes = [
        1usize, 7, 32, 63, 64, 65, 127, 128, 200, 256, 512, 777, 1024, 2048,
    ];

    for &bits_a in &sizes {
        for &bits_b in &sizes {
            for _ in 0..24 {
                let a = rng.draw(bits_a);
                let b = rng.draw(bits_b);

                // Plain gcd, both operand orders (gcd is symmetric).
                let g = gcd(&a, &b);
                assert_eq!(g, gcd_naive(&a, &b), "gcd({bits_a},{bits_b}) diverged");
                assert_eq!(gcd(&b, &a), g, "gcd asymmetric");

                // Extended gcd: exact triple must match the classical sequence,
                // and independently satisfy the Bézout identity.
                let (gx, s, t) = gcd_extended(&a, &b);
                let (gn, sn, tn) = gcd_extended_naive(&a, &b);
                assert_eq!(gx, g, "gcd_extended g disagrees with gcd");
                assert_eq!(
                    (&gx, &s, &t),
                    (&gn, &sn, &tn),
                    "gcd_extended triple diverged"
                );
                assert!(bezout_holds(&a, &b, &gx, &s, &t), "Bezout fails");
            }
        }
    }
}

#[test]
fn mod_inverse_matches_classical_over_random_sweep() {
    let mut rng = SplitMix64 {
        state: 0x0F1E_2D3C_4B5A_6978,
    };
    let sizes = [8usize, 32, 64, 65, 128, 256, 512, 1024, 2048];

    for &bits in &sizes {
        let mut checked_invertible = 0;
        let mut attempts = 0;
        while checked_invertible < 30 && attempts < 1000 {
            attempts += 1;
            // Odd modulus of the target width; random residue below it.
            let mut n = rng.draw(bits);
            if !n.is_odd() {
                n = n.add_ref(&BigUint::one());
            }
            if n.is_zero() || n.is_one() {
                continue;
            }
            let a = rng.draw(bits);

            let fast = mod_inverse(&a, &n);
            let slow = mod_inverse_naive(&a, &n);
            assert_eq!(fast, slow, "mod_inverse diverged (bits={bits})");

            if let Some(inv) = fast {
                // Independent check: a·inv ≡ 1 (mod n).
                let product = BigUint::mod_mul(&a.modulo(&n), &inv, &n);
                assert_eq!(product, BigUint::one(), "inverse does not invert");
                checked_invertible += 1;
            }
        }
    }
}

#[test]
fn jacobi_matches_division_based_over_random_sweep() {
    let mut rng = SplitMix64 {
        state: 0x2468_ACE0_1357_9BDF,
    };
    let sizes = [8usize, 32, 64, 65, 127, 128, 256, 512, 1024, 2048];
    for &bits in &sizes {
        for _ in 0..40 {
            // Odd modulus of the target width; numerator up to the same width.
            let mut n = rng.draw(bits);
            if !n.is_odd() {
                n = n.add_ref(&BigUint::one());
            }
            if n.is_zero() {
                continue;
            }
            let a = rng.draw(bits);
            assert_eq!(
                jacobi(&a, &n),
                jacobi_naive(&a, &n),
                "jacobi diverged (bits={bits})"
            );
        }
    }
    // Even and zero moduli are rejected identically.
    assert_eq!(jacobi(&BigUint::from_u64(3), &BigUint::from_u64(8)), None);
    assert_eq!(jacobi(&BigUint::one(), &BigUint::zero()), None);
    // (a/1) = 1 by the empty-product convention.
    assert_eq!(jacobi(&BigUint::from_u64(12345), &BigUint::one()), Some(1));
}

#[test]
fn gcd_family_handles_structured_corners() {
    let zero = BigUint::zero();
    let one = BigUint::one();
    let seven = BigUint::from_u64(7);

    // gcd with zero, one, self, multiples.
    assert_eq!(gcd(&zero, &zero), zero);
    assert_eq!(gcd(&seven, &zero), seven);
    assert_eq!(gcd(&zero, &seven), seven);
    assert_eq!(gcd(&seven, &one), one);
    assert_eq!(gcd(&seven, &seven), seven);

    // A large power of two against a large odd number: worst-ish for binary
    // structure, and stresses the equal-leading-digit fallback.
    let big_pow2 = {
        let mut v = one.clone();
        v.shl_bits(1500);
        v
    };
    let big_odd = big_pow2.sub_ref(&one); // 2^1500 - 1, odd
    assert_eq!(gcd(&big_pow2, &big_odd), one);
    assert_eq!(gcd(&big_pow2, &big_odd), gcd_naive(&big_pow2, &big_odd));

    // Consecutive Fibonacci-like values are the classical Euclid worst case
    // (every quotient is 1); verify Lehmer still nails the triple.
    let mut prev = BigUint::from_u64(1);
    let mut curr = BigUint::from_u64(1);
    for _ in 0..400 {
        let nextf = prev.add_ref(&curr);
        prev = curr;
        curr = nextf;
    }
    let (g, s, t) = gcd_extended(&prev, &curr);
    let (gn, sn, tn) = gcd_extended_naive(&prev, &curr);
    assert_eq!((&g, &s, &t), (&gn, &sn, &tn), "Fibonacci triple diverged");
    assert_eq!(g, one, "consecutive Fibonacci numbers are coprime");

    // Both operands equal and multi-limb: quotient is 1 then 0.
    let mut m = BigUint::from_u64(0x1234_5678_9ABC_DEF1);
    m.shl_bits(300);
    m = m.add_ref(&BigUint::from_u64(0xFEDC_BA98_7654_3210));
    assert_eq!(gcd(&m, &m), m);
    let (g, s, t) = gcd_extended(&m, &m);
    assert!(bezout_holds(&m, &m, &g, &s, &t));
}
