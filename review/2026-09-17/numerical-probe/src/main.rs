fn main() {
    use entropy::rng::{Rng, Sample};
    struct Words {
        i: usize,
    }
    impl Rng for Words {
        fn next_u32(&mut self) -> u32 {
            0
        }
        fn next_u64(&mut self) -> u64 {
            let i = self.i;
            self.i += 1;
            if i == 16 {
                1 << 14
            } else {
                0
            }
        }
    }
    println!("dense-min: {:e}", Words { i: 0 }.unit_f64_dense());
    println!("normal-min: {:?}", Words { i: 0 }.normal());
    for a in [1e-310, 1e-200, 1e-160, 1e-100, 1e-20, 0.5, 1e6, 1e10, 1e12] {
        println!(
            "beta a={a:e}: {:?}",
            rump::number_theory::regularized_incomplete_beta(0.5, a, a)
        );
    }
    for a in [1e-310, 1e-300, 1e-100, 1e-20, 1e8, 1e10, 1e12, 1e14] {
        println!("gamma {a:e}: {:.17e}", entropy::math::igamc(a, a));
    }
    for x in [2.557e305, 2.558e305, 2.559e305] {
        println!("lngamma {x:e}: {:.17e}", rump::number_theory::ln_gamma(x));
    }
    for df in [1, 2, 5, 10, 100, 10000] {
        for p in [0.75, 0.975, 0.999999] {
            println!(
                "student {df} {p}: {:?}",
                rump::number_theory::student_t_quantile(df, p)
            );
        }
    }
}
