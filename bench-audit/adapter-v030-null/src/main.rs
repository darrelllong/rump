//! Rump performance-audit adapter, v0.3.0 API names.
//!
//! Protocol: `adapter --case NAME --corpus PATH [--repeat N] [--emit time|digest|calibrate]`
//!
//! `--emit digest` prints a deterministic digest of canonicalized results and
//! no timing. `--emit calibrate` prints the repeat count reaching 20 ms.
//! `--emit time` prints `<ns_per_op> <digest>`: the digest travels with every
//! timing so the paired wrapper can refuse to emit `valid=1` on a mismatch.
//!
//! The timed region excludes corpus parsing, operand construction, calibration,
//! and all validation. Results are consumed with `black_box`.

mod cases;
mod shared;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let case = get("--case").expect("--case required");
    let corpus = get("--corpus").expect("--corpus required");
    let emit = get("--emit").unwrap_or_else(|| "time".to_string());
    let repeat: usize = get("--repeat")
        .map(|r| r.parse().expect("repeat is a number"))
        .unwrap_or(1);

    let corpus = shared::load(&corpus);
    let mut work = cases::build(&case, &corpus);

    match emit.as_str() {
        "digest" => println!("{}", work.digest()),
        "calibrate" => println!("{}", work.calibrate()),
        "time" => {
            let ns = work.time(repeat);
            println!("{ns} {}", work.digest());
        }
        other => panic!("unknown --emit {other}"),
    }
}
