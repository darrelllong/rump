//! Every code block in MANUAL.md, verbatim, one test per section.
//!
//! The manual promises that its examples are compiled and asserted on every
//! `cargo test`; this file is that promise. When an example changes, change
//! it in both places — a drifted manual fails here.

use core::num::NonZeroU64;
use rump::{
    crt_combine, gauss_reduce_weighted, gcd, gcd_extended, gcd_u64, is_probable_prime,
    is_probable_prime_bpsw, is_strong_lucas_probable_prime, jacobi, kronecker, lcm, legendre,
    lll_reduce, miller_rabin_with_bases, miller_rabin_witness, mod_inverse, mod_inverse_batch,
    mod_inverse_u64, mod_pow, mod_sqrt, mod_sqrt_prime_power, primes_below, product_tree,
    random_below, random_coprime_below, random_nonzero_below, random_probable_prime,
    rational_reconstruct, rational_reconstruct_bounded, remainder_tree, remove_factor,
    smooth_parts, valuation, BarrettContext, BigInt, BigUint, Gf2m, MontgomeryContext, PolyMod,
    PolyZ, RandomSource, Sign, SmoothnessBase, WordReciprocal,
};

#[test]
fn manual_biguint_roots_powers_bits() {
    let n = BigUint::from_u64(1_000_000);
    let (root, rem) = n.sqrt_rem();
    assert_eq!(root, BigUint::from_u64(1_000));
    assert!(rem.is_zero());
    assert!(n.is_square());
    assert!(!BigUint::from_u64(1_000_001).is_square());

    assert_eq!(
        BigUint::from_u64(1_000_000).nth_root_floor(3),
        BigUint::from_u64(100)
    );
    assert!(BigUint::from_u64(729).is_perfect_power()); // 3^6
    assert!(!BigUint::from_u64(730).is_perfect_power());

    assert_eq!(BigUint::from_u64(3).pow_u64(6), BigUint::from_u64(729));
    assert_eq!(BigUint::from_u64(0b1011_0000).popcount(), 3);
    assert_eq!(BigUint::from_u64(0b1011_0000).trailing_zeros(), Some(4));
    assert_eq!(BigUint::zero().trailing_zeros(), None);
}

#[test]
fn manual_barrett_contexts() {
    let even = BigUint::from_u64(1_000);
    let ctx = BarrettContext::new(&even).expect("modulus is at least 2");
    assert_eq!(
        ctx.mod_mul(&BigUint::from_u64(123), &BigUint::from_u64(456)),
        BigUint::from_u64(88) // 56 088 mod 1000
    );
    assert_eq!(
        ctx.mod_pow(&BigUint::from_u64(7), &BigUint::from_u64(13)),
        mod_pow(&BigUint::from_u64(7), &BigUint::from_u64(13), &even)
    );
}

#[test]
fn manual_number_theory_batch_inversion() {
    let m = BigUint::from_u64(97);
    let values = [
        BigUint::from_u64(3),
        BigUint::from_u64(10),
        BigUint::from_u64(96),
    ];
    let inverses = mod_inverse_batch(&values, &m).expect("all coprime to 97");
    for (inv, v) in inverses.iter().zip(&values) {
        assert!(BigUint::mod_mul(inv, v, &m).is_one());
    }
    assert_eq!(mod_inverse_batch(&[BigUint::from_u64(0)], &m), None);
}

#[test]
fn manual_number_theory_valuation() {
    let n = BigUint::from_u64(3_888); // 2^4 · 3^5
    assert_eq!(valuation(&n, &BigUint::from_u64(2)), 4);
    let (cofactor, exponent) = remove_factor(&n, &BigUint::from_u64(3));
    assert_eq!(exponent, 5);
    assert_eq!(cofactor, BigUint::from_u64(16));
}

#[test]
fn manual_biguint_radix_strings() {
    let n = BigUint::from_str_radix("deadbeef", 16).expect("valid hex");
    assert_eq!(n, BigUint::from_u64(0xdead_beef));
    assert_eq!(n.to_str_radix(16), "deadbeef");
    assert_eq!(n.to_string(), "3735928559");
    assert_eq!("3735928559".parse::<BigUint>(), Ok(n));

    assert_eq!(
        BigUint::from_str_radix("rump", 36),
        Some(BigUint::from_u64(1_299_409))
    );
    assert_eq!(BigUint::from_str_radix("12a", 10), None); // invalid digit

    let debt = BigInt::from_str_radix("-7", 10).expect("signed parse");
    assert_eq!(debt.to_string(), "-7");
    assert_eq!("-0".parse::<BigInt>(), Ok(BigInt::zero()));
}

#[test]
fn manual_biguint_construction_and_bytes() {
    assert!(BigUint::zero().is_zero());
    assert!(BigUint::one().is_one());

    let small = BigUint::from_u64(0xFFFF_FFFF_FFFF_FFFF);
    let wide = BigUint::from_u128(1u128 << 96);
    assert!(small < wide);

    let value = BigUint::from_be_bytes(&[0x01, 0x00]);
    assert_eq!(value, BigUint::from_u64(256));
    assert_eq!(value.to_be_bytes(), vec![0x01, 0x00]);

    // Fixed-width output pads on the left — the shape share and wire
    // serializations want; a value that does not fit panics.
    assert_eq!(value.to_be_bytes_padded(4), vec![0x00, 0x00, 0x01, 0x00]);

    // Range-pinned callers can read the low bits directly.
    let wide = BigUint::from_u128((7u128 << 64) | 9);
    assert_eq!(wide.low_u128(), (7u128 << 64) | 9);
    assert_eq!(wide.low_bits(64), BigUint::from_u64(9)); // wide mod 2^64
}

#[test]
fn manual_biguint_predicates_and_comparison() {
    let n = BigUint::from_u64(256);
    assert_eq!(n.bits(), 9);
    assert!(!n.is_odd());
    assert!(!n.is_zero());
    assert!(BigUint::from_u64(255) < n);
    assert_eq!(BigUint::zero().bits(), 0);
}

#[test]
fn manual_biguint_arithmetic() {
    let a = BigUint::from_u64(1_000);
    let b = BigUint::from_u64(37);

    assert_eq!(a.add(&b), BigUint::from_u64(1_037));
    assert_eq!(a.sub(&b), BigUint::from_u64(963));
    assert_eq!(a.mul(&b), BigUint::from_u64(37_000));
    assert_eq!(b.square(), BigUint::from_u64(1_369));
    assert_eq!(BigUint::from_u64(17).sqrt_floor(), BigUint::from_u64(4));

    let mut acc = BigUint::from_u64(1_000);
    acc.add_assign_ref(&b);
    acc.sub_assign_ref(&b);
    assert_eq!(acc, a);

    // Three-operand form: `out`'s storage is reused across calls.
    let mut out = BigUint::zero();
    out.add_into(&a, &b);
    assert_eq!(out, BigUint::from_u64(1_037));
    out.sub_into(&a, &b);
    assert_eq!(out, BigUint::from_u64(963));
}

#[test]
fn manual_biguint_shifts_and_bits() {
    let mut n = BigUint::from_u64(5);
    n.shl_bits(64);
    assert_eq!(n, BigUint::from_u128(5u128 << 64));
    n.shr_bits(64);
    assert_eq!(n, BigUint::from_u64(5));

    n.shl1();
    assert_eq!(n, BigUint::from_u64(10));
    n.shr1();
    assert_eq!(n, BigUint::from_u64(5));

    assert!(n.bit(0) && n.bit(2) && !n.bit(1));
    n.set_bit(4);
    assert_eq!(n, BigUint::from_u64(21));

    let mut x = BigUint::from_u64(0b1100);
    x.bitxor_assign(&BigUint::from_u64(0b1010));
    assert_eq!(x, BigUint::from_u64(0b0110));
}

#[test]
fn manual_biguint_division_and_reduction() {
    let n = BigUint::from_u64(1_000);
    let d = BigUint::from_u64(7);

    let (q, r) = n.div_rem(&d);
    assert_eq!(q, BigUint::from_u64(142));
    assert_eq!(r, BigUint::from_u64(6));
    assert_eq!(n.rem(&d), r);
    assert_eq!(n.rem_u64(7), 6);

    let product = BigUint::mod_mul(
        &BigUint::from_u64(123),
        &BigUint::from_u64(456),
        &BigUint::from_u64(97),
    );
    assert_eq!(product, BigUint::from_u64(22)); // 123 · 456 = 56 088 ≡ 22 (mod 97)
}

#[test]
fn manual_bigint_signed() {
    let ten = BigInt::from_biguint(BigUint::from_u64(10));
    let minus_three = BigInt::from_parts(Sign::Negative, BigUint::from_u64(3));

    // i128 construction is total: i128::MIN's magnitude is a u128.
    assert_eq!(BigInt::from_i128(-3), minus_three);
    assert_eq!(
        *BigInt::from_i128(i128::MIN).magnitude(),
        BigUint::from_u128(1u128 << 127)
    );

    assert_eq!(minus_three.sign(), Sign::Negative);
    assert_eq!(*minus_three.magnitude(), BigUint::from_u64(3));
    assert_eq!(minus_three.negated().sign(), Sign::Positive);

    assert_eq!(
        ten.add(&minus_three),
        BigInt::from_biguint(BigUint::from_u64(7))
    );
    assert_eq!(
        minus_three.sub(&ten),
        BigInt::from_parts(Sign::Negative, BigUint::from_u64(13))
    );
    assert_eq!(
        minus_three.mul_biguint(&BigUint::from_u64(4)),
        BigInt::from_parts(Sign::Negative, BigUint::from_u64(12))
    );

    // The signed ring: full product, truncated division, absolute value.
    assert_eq!(
        minus_three.mul(&ten),
        BigInt::from_parts(Sign::Negative, BigUint::from_u64(30))
    );
    // div_rem truncates toward zero; the remainder takes the dividend's sign:
    // -7 = -3·2 - 1 (floored division would say -7 = -4·2 + 1 instead).
    let minus_seven = BigInt::from_parts(Sign::Negative, BigUint::from_u64(7));
    let two = BigInt::from_biguint(BigUint::from_u64(2));
    let (q, r) = minus_seven.div_rem(&two);
    assert_eq!(q, BigInt::from_parts(Sign::Negative, BigUint::from_u64(3)));
    assert_eq!(r, BigInt::from_parts(Sign::Negative, BigUint::one()));
    assert_eq!(minus_seven.abs(), BigUint::from_u64(7));

    // −3 ≡ 8 (mod 11), in canonical range.
    assert_eq!(
        minus_three.rem_euclid(&BigUint::from_u64(11)),
        BigUint::from_u64(8)
    );

    // Zero is its own sign; from_parts normalizes it.
    assert_eq!(BigInt::zero().sign(), Sign::Zero);
    assert_eq!(
        BigInt::from_parts(Sign::Negative, BigUint::zero()).sign(),
        Sign::Zero
    );
}

#[test]
fn manual_montgomery_domain() {
    let p = BigUint::from_u64(97);
    let ctx = MontgomeryContext::new(&p).expect("97 is odd");
    assert_eq!(*ctx.modulus(), p);
    assert!(MontgomeryContext::new(&BigUint::from_u64(100)).is_none()); // even

    let a = BigUint::from_u64(5);
    let b = BigUint::from_u64(6);

    // One-shot operations convert in and out internally.
    assert_eq!(ctx.mul(&a, &b), BigUint::from_u64(30));
    assert_eq!(ctx.square(&BigUint::from_u64(9)), BigUint::from_u64(81));
    assert_eq!(ctx.pow(&a, &BigUint::from_u64(3)), BigUint::from_u64(28)); // 125 mod 97

    // Staying in the domain: encode once, multiply cheaply, decode once.
    let a_mont = ctx.encode(&a);
    let b_mont = ctx.encode(&b);
    let product_mont = ctx.mul_mont(&a_mont, &b_mont);
    assert_eq!(ctx.decode(&product_mont), BigUint::from_u64(30));

    // Loops thread one workspace through the domain operations: the same
    // values, one allocation instead of one per multiply.
    let mut ws: Vec<u64> = Vec::new();
    assert_eq!(
        ctx.mul_mont_with_workspace(&a_mont, &b_mont, &mut ws),
        product_mont
    );
    assert_eq!(
        ctx.square_mont_with_workspace(&a_mont, &mut ws),
        ctx.square_mont(&a_mont)
    );

    assert_eq!(ctx.decode(&ctx.square_mont(&a_mont)), BigUint::from_u64(25));
    assert_eq!(
        ctx.decode(&ctx.add_mont(&a_mont, &b_mont)),
        BigUint::from_u64(11)
    );
    assert_eq!(
        ctx.decode(&ctx.sub_mont(&a_mont, &b_mont)),
        BigUint::from_u64(96) // 5 − 6 ≡ −1 ≡ 96 (mod 97)
    );
    assert_eq!(ctx.decode(ctx.one_mont()), BigUint::one());

    // Reuse an encoded base across exponents.
    assert_eq!(
        ctx.pow_encoded(&a_mont, &BigUint::from_u64(3)),
        BigUint::from_u64(28)
    );
}

#[test]
fn manual_galois_fields() {
    // GF(2³) with x³ + x + 1.
    let field = Gf2m::new(BigUint::from_u64(0b1011)).expect("degree 3");
    assert_eq!(field.degree(), 3);
    assert_eq!(*field.modulus(), BigUint::from_u64(0b1011));
    assert!(Gf2m::new(BigUint::one()).is_none()); // a constant defines no field

    // (x + 1)(x² + 1) = x³ + x² + x + 1 ≡ x².
    let x_plus_1 = BigUint::from_u64(0b011);
    let x2_plus_1 = BigUint::from_u64(0b101);
    assert_eq!(field.mul(&x_plus_1, &x2_plus_1), BigUint::from_u64(0b100));
    assert_eq!(field.square(&x_plus_1), field.mul(&x_plus_1, &x_plus_1));

    // Addition is XOR; x · (x² + 1) = x³ + x ≡ 1, so they are inverses.
    assert_eq!(
        Gf2m::add(&BigUint::from_u64(0b110), &BigUint::from_u64(0b011)),
        BigUint::from_u64(0b101)
    );
    let x = BigUint::from_u64(0b010);
    assert_eq!(field.inverse(&x), Some(x2_plus_1));
    assert_eq!(field.inverse(&BigUint::zero()), None);

    // Tr(x) = 0 in GF(2³), so the half-trace solves z² + z = x.
    let z = field.half_trace(&x);
    assert_eq!(Gf2m::add(&field.square(&z), &z), x);

    // The quadratic z² + z = c has a solution exactly when Tr(c) = 0; the
    // half-trace produces it. Tr(x) = 0 and Tr(1) = 1 in GF(2³).
    assert_eq!(field.trace(&x), 0);
    assert_eq!(field.trace(&BigUint::one()), 1);

    // Squaring is a bijection; sqrt inverts it. x is the unique root of x².
    assert_eq!(field.sqrt(&BigUint::from_u64(0b100)), x);

    // x generates the order-7 multiplicative group.
    assert_eq!(field.pow(&x, &BigUint::from_u64(7)), BigUint::one());

    // Division: x² / x = x; dividing by zero is refused.
    assert_eq!(field.div(&BigUint::from_u64(0b100), &x), Some(x.clone()));
    assert_eq!(field.div(&x, &BigUint::zero()), None);

    // solve_quadratic is total across degrees — here in the AES byte field
    // GF(2⁸), where the degree is even and the half-trace does not apply.
    // c = a² + a guarantees solvability; the two roots are a and a + 1.
    let aes = Gf2m::new(BigUint::from_u64(0x11B)).expect("the AES byte field");
    let a = BigUint::from_u64(0x53);
    let c = Gf2m::add(&aes.square(&a), &a);
    let z = aes.solve_quadratic(&c).expect("solvable by construction");
    assert_eq!(Gf2m::add(&aes.square(&z), &z), c);
    assert!(z == a || z == Gf2m::add(&a, &BigUint::one()));

    // Rabin's test guards the constructor's irreducibility contract:
    // x³ + x + 1 passes, x³ + 1 = (x + 1)(x² + x + 1) fails, and the AES
    // polynomial x⁸ + x⁴ + x³ + x + 1 passes.
    assert!(Gf2m::is_irreducible(&BigUint::from_u64(0b1011)));
    assert!(!Gf2m::is_irreducible(&BigUint::from_u64(0b1001)));
    assert!(Gf2m::is_irreducible(&BigUint::from_u64(0x11B)));
}

#[test]
fn manual_number_theory_divisibility() {
    let a = BigUint::from_u64(240);
    let b = BigUint::from_u64(46);

    assert_eq!(gcd(&a, &b), BigUint::from_u64(2));
    assert_eq!(
        lcm(&BigUint::from_u64(4), &BigUint::from_u64(6)),
        BigUint::from_u64(12)
    );

    let (g, s, t) = gcd_extended(&a, &b);
    let bezout = s.mul_biguint(&a).add(&t.mul_biguint(&b));
    assert_eq!(bezout, BigInt::from_biguint(g));

    // The word-sized form answers without an allocation.
    assert_eq!(gcd_u64(240, 46), 2);
    assert_eq!(gcd_u64(0, 7), 7); // gcd(0, b) = b
}

#[test]
fn manual_number_theory_symbols() {
    // (2/9) = 1 because 9 ≡ 1 (mod 8); (3/9) = 0 by the shared factor.
    assert_eq!(
        jacobi(&BigUint::from_u64(2), &BigUint::from_u64(9)),
        Some(1)
    );
    assert_eq!(
        jacobi(&BigUint::from_u64(3), &BigUint::from_u64(9)),
        Some(0)
    );
    assert_eq!(jacobi(&BigUint::one(), &BigUint::from_u64(4)), None); // even n

    // 2 is a residue mod 7 (3² = 9 ≡ 2).
    assert_eq!(
        legendre(&BigUint::from_u64(2), &BigUint::from_u64(7)),
        Some(1)
    );

    // Kronecker handles even moduli: (5/8) = (5/2)³ = −1.
    assert_eq!(kronecker(&BigUint::from_u64(5), &BigUint::from_u64(8)), -1);
    assert_eq!(kronecker(&BigUint::one(), &BigUint::zero()), 1); // (1/0) = 1
}

#[test]
fn manual_number_theory_modular() {
    let p = BigUint::from_u64(41);

    assert_eq!(
        mod_pow(
            &BigUint::from_u64(3),
            &BigUint::from_u64(4),
            &BigUint::from_u64(5)
        ),
        BigUint::one() // 81 ≡ 1 (mod 5)
    );

    assert_eq!(
        mod_inverse(&BigUint::from_u64(3), &BigUint::from_u64(7)),
        Some(BigUint::from_u64(5)) // 3 · 5 = 15 ≡ 1 (mod 7)
    );
    assert_eq!(
        mod_inverse(&BigUint::from_u64(2), &BigUint::from_u64(4)),
        None
    );

    // The word-sized companion: extended Euclid with i128 cofactors, so it is
    // total over u64; panics only on a zero modulus.
    assert_eq!(mod_inverse_u64(3, 7), Some(5));
    assert_eq!(mod_inverse_u64(2, 4), None); // shares a factor

    let root = mod_sqrt(&BigUint::from_u64(2), &p).expect("2 is a residue mod 41");
    assert_eq!(BigUint::mod_mul(&root, &root, &p), BigUint::from_u64(2));
    assert_eq!(mod_sqrt(&BigUint::from_u64(3), &p), None); // non-residue

    // Sunzi's classic: 2 mod 3, 3 mod 5, 2 mod 7.
    let x = crt_combine(&[
        (BigUint::from_u64(2), BigUint::from_u64(3)),
        (BigUint::from_u64(3), BigUint::from_u64(5)),
        (BigUint::from_u64(2), BigUint::from_u64(7)),
    ])
    .expect("moduli are pairwise coprime");
    assert_eq!(x, BigUint::from_u64(23));
}

#[test]
fn manual_bulk_primes_word_division_estimates() {
    assert_eq!(primes_below(20), vec![2, 3, 5, 7, 11, 13, 17, 19]);

    let n = BigUint::from_u64(1_000);
    assert_eq!(n.div_rem_u64(7), (BigUint::from_u64(142), 6));
    assert_eq!(n.to_u64(), Some(1_000));
    assert_eq!(BigUint::from_u128(1u128 << 64).to_u64(), None);

    assert_eq!(BigUint::from_u64(8).to_f64_lossy(), 8.0);
    assert!((BigUint::from_u64(1_000).ln_approx() - 1_000f64.ln()).abs() < 1e-9);

    // Digit counts without the digits: the boundary cases are the powers.
    assert_eq!(BigUint::from_u64(1_000).digit_count(10), 4);
    assert_eq!(BigUint::from_u64(999).digit_count(10), 3);
    assert_eq!(BigUint::from_u64(255).digit_count(16), 2); // "ff"
    assert_eq!(BigUint::zero().digit_count(10), 1); // written "0"
}

#[test]
fn manual_division_by_an_invariant_divisor() {
    // Non-zero is the whole precondition, so the type carries it: there is no
    // error case left for the caller to handle.
    let r = WordReciprocal::new(NonZeroU64::new(1_000_003).expect("literal is non-zero"));
    assert_eq!(r.divisor(), 1_000_003);

    // The same answers as the hardware-division path. Note the argument roles:
    // `BigUint::rem_u64` takes the divisor, `WordReciprocal::rem` takes the dividend.
    assert_eq!(r.div_rem(2_000_007), (2, 1));
    assert_eq!(r.rem(2_000_006), 0);

    // Non-negative residues for signed positions.
    assert_eq!(r.rem_euclid_i64(1_000_005), 2);
    assert_eq!(r.rem_euclid_i64(-1), 1_000_002);
    assert_eq!(r.rem_euclid_i64(-1_000_003), 0);

    // Multi-limb dividends go through the same kernel, and are where it wins.
    let n = BigUint::from_u128(340_282_366_920_938_463_463_374_607_431_768_211_455);
    assert_eq!(n.rem_reciprocal(&r), n.rem_u64(1_000_003));
    assert_eq!(n.div_rem_reciprocal(&r).0, n.div_rem_u64(1_000_003).0);
}

#[test]
fn manual_prime_power_square_roots() {
    let roots = mod_sqrt_prime_power(&BigUint::from_u64(9), &BigUint::from_u64(2), 4);
    assert_eq!(
        roots,
        vec![
            BigUint::from_u64(3),
            BigUint::from_u64(5),
            BigUint::from_u64(11),
            BigUint::from_u64(13),
        ]
    );
}

#[test]
fn manual_batch_smoothness() {
    let primes = primes_below(10); // 2, 3, 5, 7
    let values = [
        BigUint::from_u64(360),    // 2^3 · 3^2 · 5 — fully smooth
        BigUint::from_u64(2 * 11), // 11 is not in the base
    ];
    let parts = smooth_parts(&values, &primes);
    assert_eq!(parts[0], BigUint::from_u64(360));
    assert_eq!(parts[1], BigUint::from_u64(2));

    let base = SmoothnessBase::new(&primes_below(10)).expect("every entry is at least two");
    assert_eq!(base.primes(), &[2, 3, 5, 7]);

    // The same answers as the free function, in batches of the caller's choosing.
    let batch = base.smooth_parts(&[BigUint::from_u64(360), BigUint::from_u64(2 * 11)]);
    assert_eq!(batch[0], BigUint::from_u64(360));
    assert_eq!(batch[1], BigUint::from_u64(2));
    assert_eq!(base.smooth_parts(&[BigUint::from_u64(360)])[0], batch[0]);

    // Smoothness is the predicate: the smooth part equals the value.
    assert!(batch[0] == BigUint::from_u64(360));
    assert!(batch[1] != BigUint::from_u64(2 * 11));

    let values = [
        BigUint::from_u64(7),
        BigUint::from_u64(11),
        BigUint::from_u64(13),
    ];
    let tree = product_tree(&values);
    assert_eq!(*tree.root().unwrap(), BigUint::from_u64(7 * 11 * 13)); // 1001
    assert_eq!(tree.len(), 3);

    let residues = remainder_tree(&tree, &BigUint::from_u64(100));
    assert_eq!(
        residues,
        vec![
            BigUint::from_u64(100 % 7),
            BigUint::from_u64(100 % 11),
            BigUint::from_u64(100 % 13),
        ]
    );
}

#[test]
fn manual_number_theory_primality() {
    assert!(is_probable_prime(&BigUint::from_u64(65_537)));
    assert!(!is_probable_prime(&BigUint::from_u64(561))); // Carmichael

    // 2047 = 23 · 89 is the smallest strong pseudoprime to base 2: the base-2
    // Miller-Rabin round is fooled (2 is not a witness), but base 3 exposes it.
    let n = BigUint::from_u64(2_047);
    assert!(!miller_rabin_witness(&n, &BigUint::from_u64(2))); // 2 is not a witness
    assert!(miller_rabin_witness(&n, &BigUint::from_u64(3))); // 3 is
    assert!(!is_probable_prime(&n)); // the full test rejects it

    // with_bases reduces each base rem n and discards {0, 1, n-1}; a set of
    // only trivial bases runs no effective round and is not reported prime.
    let composite = BigUint::from_u64(1_022_117); // 1009 × 1013, survives the sieve
    assert!(!miller_rabin_with_bases(&composite, &[1])); // 1 never testifies
    assert!(!miller_rabin_with_bases(&composite, &[2])); // 2 exposes it
}

#[test]
fn manual_number_theory_rational_reconstruction() {
    // 22/7 survives reduction mod 1009 and comes back intact.
    let m = BigUint::from_u64(1_009);
    let seven_inv = mod_inverse(&BigUint::from_u64(7), &m).expect("7 is invertible");
    let x = BigUint::mod_mul(&BigUint::from_u64(22), &seven_inv, &m);
    let (p, q) = rational_reconstruct(&x, &m).expect("22/7 is within the bounds");
    assert_eq!(p, BigInt::from_biguint(BigUint::from_u64(22)));
    assert_eq!(q, BigUint::from_u64(7));

    // Negative numerators carry their sign: −3/5 mod 1009.
    let five_inv = mod_inverse(&BigUint::from_u64(5), &m).expect("5 is invertible");
    let x = m.sub(&BigUint::mod_mul(&BigUint::from_u64(3), &five_inv, &m));
    let (p, q) = rational_reconstruct(&x, &m).expect("-3/5 is within the bounds");
    assert_eq!(p, BigInt::from_parts(Sign::Negative, BigUint::from_u64(3)));
    assert_eq!(q, BigUint::from_u64(5));

    // The bounded form is explicit about its contract.
    assert_eq!(
        rational_reconstruct_bounded(&x, &m, &BigUint::from_u64(3), &BigUint::from_u64(5)),
        Some((
            BigInt::from_parts(Sign::Negative, BigUint::from_u64(3)),
            BigUint::from_u64(5)
        ))
    );
}

#[test]
fn manual_number_theory_baillie_psw() {
    assert!(is_probable_prime_bpsw(&BigUint::from_u64(65_537)));

    // 2047 fools base 2; the Lucas stage rejects it.
    assert!(!is_probable_prime_bpsw(&BigUint::from_u64(2_047)));

    // 5459 is a strong Lucas pseudoprime; the base-2 stage rejects it.
    assert!(is_strong_lucas_probable_prime(&BigUint::from_u64(5_459)));
    assert!(!is_probable_prime_bpsw(&BigUint::from_u64(5_459)));
}

/// xorshift64: deterministic and compact. Fine for a manual; NOT a CSPRNG.
struct XorShift64(u64);

impl RandomSource for XorShift64 {
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            let word = self.0.to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
    }
}

#[test]
fn manual_random_sampling() {
    let mut rng = XorShift64(0x1234_5678_9abc_def0);
    let bound = BigUint::from_u64(1_000_003);

    let draw = random_below(&mut rng, &bound).expect("bound is non-zero");
    assert!(draw < bound);

    let nonzero = random_nonzero_below(&mut rng, &bound).expect("bound exceeds one");
    assert!(!nonzero.is_zero() && nonzero < bound);

    let unit = random_coprime_below(&mut rng, &bound, &BigUint::from_u64(30_030))
        .expect("units exist below the bound");
    assert_eq!(gcd(&unit, &BigUint::from_u64(30_030)), BigUint::one());

    let prime = random_probable_prime(&mut rng, 64).expect("width of at least 2");
    assert_eq!(prime.bits(), 64);
    assert!(is_probable_prime(&prime));

    // Degenerate bounds answer without consuming entropy.
    assert_eq!(random_below(&mut rng, &BigUint::zero()), None);
    assert_eq!(random_nonzero_below(&mut rng, &BigUint::one()), None);
    assert_eq!(random_probable_prime(&mut rng, 1), None);
}

#[test]
#[should_panic(expected = "underflow")]
fn manual_panics_unsigned_underflow() {
    let _ = BigUint::from_u64(3).sub(&BigUint::from_u64(5));
}

#[test]
#[should_panic(expected = "division by zero")]
fn manual_panics_division_by_zero() {
    let _ = BigUint::from_u64(3).div_rem(&BigUint::zero());
}

#[test]
fn manual_ordinary_code_sorting() {
    /// Sort in place by repeated adjacent exchange — pedagogical, not fast.
    fn bubble_sort(values: &mut [BigInt]) {
        for pass in 1..values.len() {
            for i in 0..values.len() - pass {
                if values[i] > values[i + 1] {
                    values.swap(i, i + 1);
                }
            }
        }
    }

    // A signed value is built from an unsigned magnitude; negation is a method.
    let big = |v: u64| BigInt::from_biguint(BigUint::from_u64(v));
    let neg = |v: u64| big(v).negated();

    // One operand wider than any machine word: 2^100 + 7.
    let mut wide = BigUint::one();
    wide.shl_bits(100);
    let wide = BigInt::from_biguint(wide.add(&BigUint::from_u64(7)));

    let mut values = vec![big(251), neg(40), big(0), wide.clone(), neg(3), big(17)];
    bubble_sort(&mut values);
    assert_eq!(
        values,
        vec![neg(40), neg(3), big(0), big(17), big(251), wide]
    );
}

#[test]
fn manual_polynomials() {
    // (x + 1)(x + 2) = x^2 + 3x + 2, over ℤ.
    let a = PolyZ::from_i64_slice(&[1, 1]); // x + 1
    let b = PolyZ::from_i64_slice(&[2, 1]); // x + 2
    assert_eq!(a.mul(&b), PolyZ::from_i64_slice(&[2, 3, 1]));

    // Exact division over ℤ: (x^2 + 3x + 2) / (x + 1) = x + 2, no remainder.
    let (q, r) = a.mul(&b).div_rem(&a).expect("x + 1 divides");
    assert_eq!(q, b); // x + 2
    assert!(r.is_zero());
    // A leading coefficient that does not divide has no integer quotient.
    assert_eq!(
        PolyZ::from_i64_slice(&[1, 0, 1]).div_rem(&PolyZ::from_i64_slice(&[1, 2])),
        None
    );

    // Evaluation and derivative.
    assert_eq!(
        a.mul(&b).evaluate(&BigInt::from_i64(3)),
        BigInt::from_i64(20)
    ); // 9+9+2
    assert_eq!(a.mul(&b).derivative(), PolyZ::from_i64_slice(&[3, 2])); // 2x + 3

    // content and primitive part of 6x^2 + 9x + 15.
    let c = PolyZ::from_i64_slice(&[15, 9, 6]);
    assert_eq!(c.content(), BigInt::from_i64(3));
    assert_eq!(c.primitive_part(), PolyZ::from_i64_slice(&[5, 3, 2]));

    // Over ℤ/7ℤ: (x - 1)(x - 2) and (x - 2)(x - 3) share x - 2.
    let p = BigUint::from_u64(7);
    let f = PolyMod::from_poly_z(&PolyZ::from_i64_slice(&[2, -3, 1]), &p); // x^2 - 3x + 2
    let g = PolyMod::from_poly_z(&PolyZ::from_i64_slice(&[6, -5, 1]), &p); // x^2 - 5x + 6
    let shared = PolyMod::from_poly_z(&PolyZ::from_i64_slice(&[-2, 1]), &p); // x - 2
    assert_eq!(f.gcd(&g), shared);

    // disc(x^2 + 5x + 6) = 25 - 24 = 1.
    let quad = PolyZ::from_i64_slice(&[6, 5, 1]);
    assert_eq!(quad.discriminant(), BigInt::from_i64(1));

    // (x-1)(x-2) and (x-1)(x-3) share (x-1): resultant zero.
    let u = PolyZ::from_i64_slice(&[2, -3, 1]); // x^2 - 3x + 2
    let v = PolyZ::from_i64_slice(&[3, -4, 1]); // x^2 - 4x + 3
    assert_eq!(u.resultant(&v), BigInt::zero());

    // A perfect square has discriminant zero: (x-1)^2 = x^2 - 2x + 1.
    assert_eq!(
        PolyZ::from_i64_slice(&[1, -2, 1]).discriminant(),
        BigInt::zero()
    );

    struct Lcg(u64);
    impl RandomSource for Lcg {
        fn fill_bytes(&mut self, d: &mut [u8]) {
            for b in d {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
                *b = (self.0 >> 33) as u8;
            }
        }
    }
    let p = BigUint::from_u64(7);
    let mut rng = Lcg(0x1234_5678);

    // x^2 - 1 = (x - 1)(x + 1) mod 7 — two roots.
    let f = PolyMod::from_poly_z(&PolyZ::from_i64_slice(&[-1, 0, 1]), &p);
    let mut roots = f.roots(&mut rng);
    roots.sort();
    assert_eq!(roots, vec![BigUint::from_u64(1), BigUint::from_u64(6)]); // 6 ≡ -1

    // x^2 + 1 is irreducible mod 7 (−1 is a non-residue), so no roots.
    let g = PolyMod::from_poly_z(&PolyZ::from_i64_slice(&[1, 0, 1]), &p);
    assert!(g.is_irreducible());
    assert!(g.roots(&mut rng).is_empty());

    // factor returns monic irreducibles with multiplicities:
    // x^3 + x = x · (x^2 + 1) over F_7.
    let h = PolyMod::from_poly_z(&PolyZ::from_i64_slice(&[0, 1, 0, 1]), &p);
    let mut fs = h.factor(&mut rng);
    fs.sort_by_key(|(fac, _)| fac.degree());
    assert_eq!(fs.len(), 2);
    assert_eq!(fs[0].0.degree(), Some(1)); // x
    assert_eq!(fs[1].0.degree(), Some(2)); // x^2 + 1
    assert!(fs.iter().all(|(_, e)| *e == 1));
}

#[test]
fn manual_lattice() {
    let row = |xs: &[i64]| xs.iter().map(|&x| BigInt::from_i64(x)).collect::<Vec<_>>();
    // A badly skewed basis for a 2-D lattice.
    let mut basis = vec![row(&[201, 37]), row(&[1648, 297])];
    lll_reduce(&mut basis);
    // Reduction returns short vectors spanning the same lattice.
    assert_eq!(basis, vec![row(&[1, 32]), row(&[40, 1])]);

    // A skew-12 lattice, reduced under the matching diagonal form.
    let reduced = gauss_reduce_weighted([[1024, 0], [37, 1]], [1, 12]).expect("a valid basis");
    // Same lattice: the determinant is preserved up to sign.
    let det = |b: [[i128; 2]; 2]| b[0][0] * b[1][1] - b[0][1] * b[1][0];
    assert_eq!(det(reduced).abs(), det([[1024, 0], [37, 1]]).abs());

    // Weights reorder what counts as short: under a heavy x-weight the
    // y-axis vector wins, and under a heavy y-weight the x-axis vector does.
    assert_eq!(
        gauss_reduce_weighted([[1, 0], [0, 1]], [100, 1]),
        Some([[0, 1], [1, 0]])
    );
    assert_eq!(
        gauss_reduce_weighted([[1, 0], [0, 1]], [1, 100]),
        Some([[1, 0], [0, 1]])
    );
}

#[test]
fn manual_polynomials_quotient_rings_and_lifting() {
    struct Lcg2(u64);
    impl RandomSource for Lcg2 {
        fn fill_bytes(&mut self, d: &mut [u8]) {
            for b in d {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
                *b = (self.0 >> 33) as u8;
            }
        }
    }
    let mut rng = Lcg2(0x2024_1111);

    // Reduction rem the monic x^2 + 1 is a ring homomorphism.
    let f = PolyZ::from_i64_slice(&[1, 0, 1]);
    let a = PolyZ::from_i64_slice(&[3, 2, 5]); // 5x^2 + 2x + 3
    let b = PolyZ::from_i64_slice(&[1, 7]); // 7x + 1
    assert_eq!(
        a.mul(&b).rem_monic(&f),
        a.rem_monic(&f).mul(&b.rem_monic(&f)).rem_monic(&f)
    );

    // The product tree in Z[x]/(f) agrees with the fold, reduced once.
    let x = PolyZ::from_i64_slice(&[0, 1]);
    let factors = [a.clone(), b.clone(), x.clone()];
    assert_eq!(
        PolyZ::product_mod_monic(&factors, &f),
        a.mul(&b).mul(&x).rem_monic(&f)
    );

    // Balanced base-1000 expansion of 1234567890: digits in (-500, 500].
    let n = BigInt::from_i64(1_234_567_890);
    let base = BigUint::from_u64(1_000);
    let g = PolyZ::balanced_base_expansion(&n, &base, 3);
    assert_eq!(g, PolyZ::from_i64_slice(&[-110, -432, 235, 1]));
    assert_eq!(g.evaluate(&BigInt::from_i64(1_000)), n); // exact, by construction

    // Homogenization of x^2 + 1 at (2x, 3): 3^2·1 + (2x)^2 = 4x^2 + 9.
    assert_eq!(
        f.homogeneous_substitution(
            &PolyZ::from_i64_slice(&[0, 2]),
            &PolyZ::from_i64_slice(&[3])
        ),
        PolyZ::from_i64_slice(&[9, 0, 4])
    );

    // Square roots of 2 rem 7^3 = 343, lifted from ±3 rem 7.
    let sqrt2 = PolyZ::from_i64_slice(&[-2, 0, 1]).roots_mod_prime_power(
        &BigUint::from_u64(7),
        3,
        &mut rng,
    );
    assert_eq!(sqrt2, vec![BigUint::from_u64(108), BigUint::from_u64(235)]);
    assert_eq!(108u64 * 108 % 343, 2);

    // A symmetric lift recovers the integer polynomial it came from.
    let wide = BigUint::from_u64(2).pow_u64(96);
    assert_eq!(PolyMod::from_poly_z(&a, &wide).symmetric_lift(), a);
}
