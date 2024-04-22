use diol::prelude::*;

fn main() -> std::io::Result<()> {
    let mut bench = Bench::new(BenchConfig::from_args()?);
    bench.register(slice_times_two, [4, 8, 16, 128, 1024]);
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
