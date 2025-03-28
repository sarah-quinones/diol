use diol::prelude::*;
use eyre::Result;

fn main() -> Result<()> {
    let bench = Bench::new(BenchConfig::from_args()?);
    bench.register_many(
        "slice × 2",
        list![slice_times_two, slice_times_two_autovec],
        [4, 8, 16, 128, 1024, 2048, 4096].map(PlotArg),
    );
    bench.run()?;
    Ok(())
}

fn slice_times_two(bencher: Bencher, PlotArg(len): PlotArg) {
    let mut v = vec![0.0_f64; len];
    bencher.bench(|| {
        for x in &mut v {
            *x *= 2.0;
        }
        black_box(&mut v);
    });
}

fn slice_times_two_autovec(bencher: Bencher, PlotArg(len): PlotArg) {
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
