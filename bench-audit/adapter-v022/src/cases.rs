//! The workload set. This is the only file that differs between the two
//! revision adapters, and only where the 0.3.0 rename forces it.
//!
//! v0.2.2 names used here: the flat crate root, `add_ref`/`sub_ref`/`mul_ref`/
//! `square_ref`, `BarrettCtx`/`MontgomeryCtx`, `encode`/`mul_mont`, `sqrt_mod`,
//! `PolyModP`. The workload bodies are otherwise identical to the v0.3.0
//! adapter's.

use crate::shared::{calibrate, small_primes, timed, Corpus, Digest, SplitMix};
use std::hint::black_box;

use rump::{
    gcd, gcd_extended, jacobi, lll_reduce, mod_inverse, mod_inverse_batch, mod_pow,
    product_tree, random_below, random_coprime_below, random_probable_prime, remainder_tree,
    smooth_parts, sqrt_mod, BarrettCtx, BigInt, BigUint, Gf2m, MontgomeryCtx, PolyZ, Rng,
};

/// Fixed seed, so a case's draws are a property of the case and not of the run.
const SEED: u64 = 0x5eed_0fa4_d17d;

/// Bridges the shared generator to this revision's random-source trait.
struct Adapter(SplitMix);

impl Rng for Adapter {
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill(dest);
    }
}

/// A `width`-bit integer drawn from the fixed seed, independent of any corpus
/// value. Deterministic, so both revisions reduce the same modulus.
fn independent_modulus(width: usize) -> BigUint {
    let mut draw = SplitMix(SEED);
    let digits = width.div_ceil(4).max(2);
    let mut hexits = String::with_capacity(digits);
    while hexits.len() < digits {
        hexits.push_str(&format!("{:016x}", draw.next_u64()));
    }
    hexits.truncate(digits);
    // Force the top hexit high so the value really is `width` bits wide, and
    // the last bit set so it is odd and shares no factor two with a leaf.
    let mut bytes: Vec<char> = hexits.chars().collect();
    bytes[0] = 'f';
    let last = bytes.len() - 1;
    bytes[last] = 'd';
    hex(&bytes.into_iter().collect::<String>())
}

/// A built workload: operands already parsed, ready to time.
pub struct Work {
    /// Operations per traversal, for the ns/op conversion.
    ops: usize,
    run: Box<dyn FnMut()>,
    /// Canonical result strings, produced outside the timed region.
    results: Vec<String>,
}

impl Work {
    pub fn digest(&self) -> String {
        let mut d = Digest::new();
        for r in &self.results {
            d.add(r);
        }
        d.finish()
    }
    pub fn results(&self) -> &[String] {
        &self.results
    }
    pub fn calibrate(&mut self) -> usize {
        calibrate(20.0, &mut *self.run)
    }
    pub fn time(&mut self, repeat: usize) -> f64 {
        let ops = self.ops;
        timed(repeat, ops, &mut *self.run)
    }
}

fn hex(s: &str) -> BigUint {
    BigUint::from_str_radix(s, 16).expect("corpus operand is hex")
}

/// Operands as (a, b) pairs.
fn pairs(c: &Corpus) -> Vec<(BigUint, BigUint)> {
    c.items
        .chunks(2)
        .filter(|ch| ch.len() == 2)
        .map(|ch| (hex(&ch[0]), hex(&ch[1])))
        .collect()
}

fn singles(c: &Corpus) -> Vec<BigUint> {
    c.items.iter().map(|s| hex(s)).collect()
}

/// Triples as (base, exponent, modulus).
fn triples(c: &Corpus) -> Vec<(BigUint, BigUint, BigUint)> {
    c.items
        .chunks(3)
        .filter(|ch| ch.len() == 3)
        .map(|ch| (hex(&ch[0]), hex(&ch[1]), hex(&ch[2])))
        .collect()
}

fn polys(c: &Corpus) -> Vec<PolyZ> {
    c.groups
        .iter()
        .map(|g| {
            PolyZ::new(
                g.iter()
                    .map(|s| BigInt::from_biguint(hex(s)))
                    .collect::<Vec<_>>(),
            )
        })
        .collect()
}

pub fn build(case: &str, c: &Corpus) -> Work {
    match case {
        "int_add" => {
            let ops = pairs(c);
            let results = ops.iter().map(|(a, b)| a.add_ref(b).to_str_radix(16)).collect();
            let data = ops;
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, b) in &data {
                        black_box(a.add_ref(b));
                    }
                }),
            }
        }
        "int_sub" => {
            let ops: Vec<(BigUint, BigUint)> = pairs(c)
                .into_iter()
                .map(|(a, b)| if a >= b { (a, b) } else { (b, a) })
                .collect();
            let results = ops.iter().map(|(a, b)| a.sub_ref(b).to_str_radix(16)).collect();
            let data = ops;
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, b) in &data {
                        black_box(a.sub_ref(b));
                    }
                }),
            }
        }
        "int_mul" => {
            let data = pairs(c);
            let results = data.iter().map(|(a, b)| a.mul_ref(b).to_str_radix(16)).collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, b) in &data {
                        black_box(a.mul_ref(b));
                    }
                }),
            }
        }
        "int_square" => {
            let data: Vec<BigUint> = pairs(c).into_iter().map(|(a, _)| a).collect();
            let results = data.iter().map(|a| a.square_ref().to_str_radix(16)).collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for a in &data {
                        black_box(a.square_ref());
                    }
                }),
            }
        }
        "int_div_rem" => {
            let data: Vec<(BigUint, BigUint)> = pairs(c)
                .into_iter()
                .filter(|(_, b)| !b.is_zero())
                .collect();
            let results = data
                .iter()
                .map(|(a, b)| {
                    let (q, r) = a.div_rem(b);
                    format!("{} {}", q.to_str_radix(16), r.to_str_radix(16))
                })
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, b) in &data {
                        black_box(a.div_rem(b));
                    }
                }),
            }
        }
        "int_to_hex" => {
            let data: Vec<BigUint> = pairs(c).into_iter().map(|(a, _)| a).collect();
            let results = data.iter().map(|a| a.to_str_radix(16)).collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for a in &data {
                        black_box(a.to_str_radix(16));
                    }
                }),
            }
        }
        "int_to_dec" => {
            let data: Vec<BigUint> = pairs(c).into_iter().map(|(a, _)| a).collect();
            let results = data.iter().map(|a| a.to_str_radix(10)).collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for a in &data {
                        black_box(a.to_str_radix(10));
                    }
                }),
            }
        }
        "nt_gcd" => {
            let data = pairs(c);
            let results = data.iter().map(|(a, b)| gcd(a, b).to_str_radix(16)).collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, b) in &data {
                        black_box(gcd(a, b));
                    }
                }),
            }
        }
        "nt_gcd_extended" => {
            let data = pairs(c);
            let results = data
                .iter()
                .map(|(a, b)| {
                    let (g, _, _) = gcd_extended(a, b);
                    g.to_str_radix(16)
                })
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, b) in &data {
                        black_box(gcd_extended(a, b));
                    }
                }),
            }
        }
        "nt_jacobi" => {
            let data: Vec<(BigUint, BigUint)> = pairs(c)
                .into_iter()
                .map(|(a, b)| {
                    let mut n = b;
                    if !n.is_odd() {
                        n = n.add_ref(&BigUint::one());
                    }
                    (a, n)
                })
                .collect();
            let results = data
                .iter()
                .map(|(a, n)| format!("{:?}", jacobi(a, n)))
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, n) in &data {
                        black_box(jacobi(a, n));
                    }
                }),
            }
        }
        "mod_pow" => {
            let data = triples(c);
            let results = data
                .iter()
                .map(|(b, e, m)| mod_pow(b, e, m).to_str_radix(16))
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (b, e, m) in &data {
                        black_box(mod_pow(b, e, m));
                    }
                }),
            }
        }
        "mod_inverse" => {
            let data = pairs(c);
            let results = data
                .iter()
                .map(|(a, m)| match mod_inverse(a, m) {
                    Some(i) => i.to_str_radix(16),
                    None => "none".to_string(),
                })
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, m) in &data {
                        black_box(mod_inverse(a, m));
                    }
                }),
            }
        }
        // v0.2.2 calls the function `sqrt_mod`; the case keeps the canonical
        // name so a cell means the same workload under both revisions.
        "mod_sqrt" => {
            let prime = hex(&c.items[0]);
            let data: Vec<BigUint> = c.items[1..].iter().map(|s| hex(s)).collect();
            let results = data
                .iter()
                .map(|a| match sqrt_mod(a, &prime) {
                    Some(r) => r.to_str_radix(16),
                    None => "none".to_string(),
                })
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for a in &data {
                        black_box(sqrt_mod(a, &prime));
                    }
                }),
            }
        }
        // Constructor cost, measured on its own.
        "barrett_new" => {
            let data: Vec<BigUint> = singles(c);
            let results = data
                .iter()
                .map(|m| {
                    BarrettCtx::new(m)
                        .map(|ctx| ctx.modulus().to_str_radix(16))
                        .unwrap_or_else(|| "err".to_string())
                })
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for m in &data {
                        black_box(BarrettCtx::new(m));
                    }
                }),
            }
        }
        "montgomery_new" => {
            let data: Vec<BigUint> = singles(c);
            let results = data
                .iter()
                .map(|m| {
                    MontgomeryCtx::new(m)
                        .map(|ctx| ctx.modulus().to_str_radix(16))
                        .unwrap_or_else(|| "err".to_string())
                })
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for m in &data {
                        black_box(MontgomeryCtx::new(m));
                    }
                }),
            }
        }
        // Reuse cost with a shared context: the setup is outside the timed
        // region, so this is the steady-state kernel.
        "barrett_mod_mul" => {
            let moduli: Vec<BigUint> = singles(c);
            let contexts: Vec<BarrettCtx> = moduli
                .iter()
                .filter_map(|m| BarrettCtx::new(m))
                .collect();
            let operands: Vec<(BigUint, BigUint)> = moduli
                .iter()
                .map(|m| (m.sub_ref(&BigUint::one()), m.sub_ref(&BigUint::from_u64(2))))
                .collect();
            let results = contexts
                .iter()
                .zip(&operands)
                .map(|(ctx, (a, b))| ctx.mul_mod(a, b).to_str_radix(16))
                .collect();
            let data: Vec<(BarrettCtx, BigUint, BigUint)> = contexts
                .into_iter()
                .zip(operands)
                .map(|(ctx, (a, b))| (ctx, a, b))
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (ctx, a, b) in &data {
                        black_box(ctx.mul_mod(a, b));
                    }
                }),
            }
        }
        "montgomery_mul_residue" => {
            let moduli: Vec<BigUint> = singles(c);
            let mut prepared = Vec::new();
            let mut results = Vec::new();
            for m in &moduli {
                let Some(ctx) = MontgomeryCtx::new(m) else {
                    continue;
                };
                let a = ctx.encode(&m.sub_ref(&BigUint::one()));
                let b = ctx.encode(&m.sub_ref(&BigUint::from_u64(2)));
                let product = ctx.mul_mont(&a, &b);
                results.push(ctx.decode(&product).to_str_radix(16));
                prepared.push((ctx, a, b));
            }
            Work {
                ops: prepared.len(),
                results,
                run: Box::new(move || {
                    for (ctx, a, b) in &prepared {
                        black_box(ctx.mul_mont(a, b));
                    }
                }),
            }
        }
        "poly_mul" => {
            let data: Vec<(PolyZ, PolyZ)> = polys(c)
                .chunks(2)
                .filter(|ch| ch.len() == 2)
                .map(|ch| (ch[0].clone(), ch[1].clone()))
                .collect();
            let results = data
                .iter()
                .map(|(f, g)| {
                    let p = f.mul(g);
                    format!("{:?}", p.degree())
                })
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (f, g) in &data {
                        black_box(f.mul(g));
                    }
                }),
            }
        }
        "gf2m_mul" => {
            let field = Gf2m::new(hex(&c.items[0])).expect("irreducible field polynomial");
            let elements: Vec<BigUint> = c.items[1..].iter().map(|s| hex(s)).collect();
            let data: Vec<(BigUint, BigUint)> = elements
                .chunks(2)
                .filter(|ch| ch.len() == 2)
                .map(|ch| (ch[0].clone(), ch[1].clone()))
                .collect();
            let results = data
                .iter()
                .map(|(a, b)| field.mul(a, b).to_str_radix(16))
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, b) in &data {
                        black_box(field.mul(a, b));
                    }
                }),
            }
        }
        "gf2m_square" => {
            let field = Gf2m::new(hex(&c.items[0])).expect("irreducible field polynomial");
            let data: Vec<BigUint> = c.items[1..].iter().map(|s| hex(s)).collect();
            let results = data.iter().map(|a| field.square(a).to_str_radix(16)).collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for a in &data {
                        black_box(field.square(a));
                    }
                }),
            }
        }
        "gf2m_inverse" => {
            let field = Gf2m::new(hex(&c.items[0])).expect("irreducible field polynomial");
            let data: Vec<BigUint> = c.items[1..]
                .iter()
                .map(|s| hex(s))
                .filter(|a| !a.is_zero())
                .collect();
            let results = data
                .iter()
                .map(|a| match field.inverse(a) {
                    Some(i) => i.to_str_radix(16),
                    None => "none".to_string(),
                })
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for a in &data {
                        black_box(field.inverse(a));
                    }
                }),
            }
        }
        "gf2m_sqrt" => {
            let field = Gf2m::new(hex(&c.items[0])).expect("irreducible field polynomial");
            let data: Vec<BigUint> = c.items[1..].iter().map(|s| hex(s)).collect();
            let results = data.iter().map(|a| field.sqrt(a).to_str_radix(16)).collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for a in &data {
                        black_box(field.sqrt(a));
                    }
                }),
            }
        }
        "lattice_lll" => {
            // One basis per group; the reduction is in place, so each timed
            // traversal works on a fresh clone. Cloning is inside the loop
            // because the operation consumes its input, and the same clone cost
            // is charged to both revisions identically.
            let dimension: usize = c
                .note("dimension")
                .and_then(|d| d.parse().ok())
                .expect("lattice corpus declares a dimension");
            let bases: Vec<Vec<Vec<BigInt>>> = c
                .groups
                .iter()
                .map(|g| {
                    g.chunks(dimension)
                        .map(|row| {
                            row.iter()
                                .map(|s| BigInt::from_biguint(hex(s)))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            let results = bases
                .iter()
                .map(|b| {
                    let mut work = b.clone();
                    lll_reduce(&mut work);
                    let mut out = String::new();
                    for row in &work {
                        for entry in row {
                            out.push_str(&entry.to_str_radix(16));
                            out.push(',');
                        }
                        out.push(';');
                    }
                    out
                })
                .collect();
            Work {
                ops: bases.len(),
                results,
                run: Box::new(move || {
                    for b in &bases {
                        let mut work = b.clone();
                        lll_reduce(&mut work);
                        black_box(&work);
                    }
                }),
            }
        }
        "batch_mod_inverse" => {
            let values: Vec<BigUint> = singles(c);
            // A modulus coprime to the batch: the largest value plus two, made
            // odd, so inversion succeeds for the whole batch.
            let modulus = values
                .iter()
                .max()
                .expect("non-empty batch")
                .add_ref(&BigUint::from_u64(2));
            let results = match mod_inverse_batch(&values, &modulus) {
                Some(v) => v.iter().map(|i| i.to_str_radix(16)).collect(),
                None => vec!["none".to_string()],
            };
            Work {
                ops: 1,
                results,
                run: Box::new(move || {
                    black_box(mod_inverse_batch(&values, &modulus));
                }),
            }
        }
        "batch_smooth_parts" => {
            let values: Vec<BigUint> = singles(c);
            let primes = small_primes(512);
            let results = smooth_parts(&values, &primes)
                .iter()
                .map(|s| s.to_str_radix(16))
                .collect();
            Work {
                ops: 1,
                results,
                run: Box::new(move || {
                    black_box(smooth_parts(&values, &primes));
                }),
            }
        }
        "rand_below" => {
            let bounds: Vec<BigUint> = singles(c);
            let results = {
                let mut rng = Adapter(SplitMix(SEED));
                bounds
                    .iter()
                    .map(|b| match random_below(&mut rng, b) {
                        Some(v) => v.to_str_radix(16),
                        None => "none".to_string(),
                    })
                    .collect()
            };
            Work {
                ops: bounds.len(),
                results,
                run: Box::new(move || {
                    let mut rng = Adapter(SplitMix(SEED));
                    for b in &bounds {
                        black_box(random_below(&mut rng, b));
                    }
                }),
            }
        }
        "rand_coprime_below" => {
            let bounds: Vec<BigUint> = singles(c);
            // 2·3·5·7·11·13. Only φ(30030)/30030 ≈ 0.19 of the residues are
            // coprime to it, so roughly four draws in five are rejected — the
            // rejection loop is what this case is here to measure.
            let coprime_to = BigUint::from_u64(30030);
            let results = {
                let mut rng = Adapter(SplitMix(SEED));
                bounds
                    .iter()
                    .map(|b| match random_coprime_below(&mut rng, b, &coprime_to) {
                        Some(v) => v.to_str_radix(16),
                        None => "none".to_string(),
                    })
                    .collect()
            };
            Work {
                ops: bounds.len(),
                results,
                run: Box::new(move || {
                    let mut rng = Adapter(SplitMix(SEED));
                    for b in &bounds {
                        black_box(random_coprime_below(&mut rng, b, &coprime_to));
                    }
                }),
            }
        }
        "rand_probable_prime" => {
            // The bit width comes from the corpus note, not from the operands:
            // this sampler takes a width, and the corpus is here to name it.
            let bits: usize = c
                .note("bits")
                .and_then(|b| b.parse().ok())
                .expect("bound corpus declares a width");
            let results = {
                let mut rng = Adapter(SplitMix(SEED));
                vec![match random_probable_prime(&mut rng, bits) {
                    Some(v) => v.to_str_radix(16),
                    None => "none".to_string(),
                }]
            };
            Work {
                ops: 1,
                results,
                run: Box::new(move || {
                    let mut rng = Adapter(SplitMix(SEED));
                    black_box(random_probable_prime(&mut rng, bits));
                }),
            }
        }
        "batch_product_tree" => {
            let values: Vec<BigUint> = singles(c);
            let tree = product_tree(&values);
            let results = tree
                .last()
                .map(|root| root.iter().map(|n| n.to_str_radix(16)).collect())
                .unwrap_or_else(|| vec!["empty".to_string()]);
            Work {
                ops: 1,
                results,
                run: Box::new(move || {
                    black_box(product_tree(&values));
                }),
            }
        }
        "batch_remainder_tree" => {
            let values: Vec<BigUint> = singles(c);
            let tree = product_tree(&values);
            // A modulus drawn independently of the batch, at the width of the
            // product. `root + k` for small `k` is what Bernstein's algorithm
            // is *not* used for: it makes every remainder equal to `k`, which
            // both trivialises the arithmetic and reduces the digest to a
            // restatement of the batch size.
            let width = tree.last().and_then(|r| r.first()).map_or(64, |r| r.bits());
            let modulus = independent_modulus(width);
            let results = remainder_tree(&tree, &modulus)
                .iter()
                .map(|r| r.to_str_radix(16))
                .collect();
            Work {
                ops: 1,
                results,
                run: Box::new(move || {
                    black_box(remainder_tree(&tree, &modulus));
                }),
            }
        }
        other => panic!("unknown case {other}"),
    }
}
