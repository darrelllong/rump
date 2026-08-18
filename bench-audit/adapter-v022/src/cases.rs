//! The workload set. This is the only file that differs between the two
//! revision adapters, and only where the 0.3.0 rename forces it.
//!
//! v0.2.2 names used here: the flat crate root, `add_ref`/`sub_ref`/`mul_ref`/
//! `square_ref`, `BarrettCtx`/`MontgomeryCtx`, `encode`/`mul_mont`, `sqrt_mod`,
//! `PolyModP`. The workload bodies are otherwise identical to the v0.3.0
//! adapter's.

use crate::shared::{calibrate, timed, Corpus, Digest};
use std::hint::black_box;

use rump::{
    gcd, gcd_extended, jacobi, mod_inverse, mod_pow, sqrt_mod, BarrettCtx, BigInt, BigUint,
    MontgomeryCtx, PolyZ,
};

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
        "sqrt_mod" => {
            let data: Vec<(BigUint, BigUint)> = pairs(c)
                .into_iter()
                .map(|(a, m)| {
                    let mut p = m;
                    if !p.is_odd() {
                        p = p.add_ref(&BigUint::one());
                    }
                    (a, p)
                })
                .collect();
            let results = data
                .iter()
                .map(|(a, p)| match sqrt_mod(a, p) {
                    Some(r) => r.to_str_radix(16),
                    None => "none".to_string(),
                })
                .collect();
            Work {
                ops: data.len(),
                results,
                run: Box::new(move || {
                    for (a, p) in &data {
                        black_box(sqrt_mod(a, p));
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
        other => panic!("unknown case {other}"),
    }
}
