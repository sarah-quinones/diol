use diol::prelude::*;
use eyre::Result;

fn main() -> Result<()> {
    let bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(
        "slice × 2",
        list![
            // the benchmark name can be automatically deduced..
            slice_times_two,
            // or optionally provided.
            slice_times_two_autovec.with_name("times two autovec"),
        ],
        [4, 8, 16, 128, 1024],
    );
    bench.run()?;
    Ok(())
}

fn slice_times_two(bencher: Bencher, len: usize) {
    let mut v = vec![0.0_f64; len];
    bencher.bench(|| {
        for x in &mut v {
            *x *= 2.0;
        }
        black_box(&mut v);
    });
}

fn slice_times_two_autovec(bencher: Bencher, len: usize) {
    let mut v = vec![0.0_f64; len];
    let arch = pulp::Arch::new();
    bencher.bench(|| {
        arch.dispatch(
            #[inline(always)]
            || {
                for x in &mut v {
                    *x *= 2.0;
                }
            },
        );
        black_box(&mut v);
    });
}
