`diol` is a benchmarking library for rust.
# getting started
add the following to your `Cargo.toml`.
```notcode
[dev-dependencies]
diol = "0.6.0"
[[bench]]
name = "my_benchmark"
harness = false
```
then in `benches/my_benchmark.rs`.
```rust
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
```
run the benchmark with `cargo bench`, or `cargo bench --bench my_benchmark` if you have multiple
benchmarks you can also pass in benchmark options using `cargo bench --bench my_benchmark --
[OPTIONS...]`
```
╭─────────────────┬──────┬───────────┬───────────┬───────────┬───────────╮
│ benchmark       │ args │   fastest │    median │      mean │    stddev │
├─────────────────┼──────┼───────────┼───────────┼───────────┼───────────┤
│ slice_times_two │    4 │  29.61 ns │  34.38 ns │  34.83 ns │   1.62 ns │
├─────────────────┼──────┼───────────┼───────────┼───────────┼───────────┤
│ slice_times_two │    8 │  44.17 ns │  53.04 ns │  53.32 ns │   3.25 ns │
├─────────────────┼──────┼───────────┼───────────┼───────────┼───────────┤
│ slice_times_two │   16 │  93.66 ns │ 107.91 ns │ 108.13 ns │   4.11 ns │
├─────────────────┼──────┼───────────┼───────────┼───────────┼───────────┤
│ slice_times_two │  128 │ 489.97 ns │ 583.59 ns │ 585.28 ns │  33.15 ns │
├─────────────────┼──────┼───────────┼───────────┼───────────┼───────────┤
│ slice_times_two │ 1024 │   3.77 µs │   4.51 µs │   4.53 µs │ 173.44 ns │
╰─────────────────┴──────┴───────────┴───────────┴───────────┴───────────╯
```

# dependencies

the plotters dependency requires the `pkg-config`, `freetype` and `fontconfig`.  
to install on Ubuntu, you can use the following command.

```
sudo apt install pkg-config libfreetype6-dev libfontconfig1-dev
```
