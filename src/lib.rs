use dyn_clone::DynClone;
use equator::assert;
use plotters::coord::ranged1d::{AsRangedCoord, ValueFormatter};
use plotters::element::PointCollection;
use plotters::prelude::*;
use plotters::style::full_palette::*;
use regex::Regex;
use serde::Deserialize;
use std::any::TypeId;
use std::fmt;
use std::path::PathBuf;
use std::process::Command;
use std::{any::Any, time::Duration};

use traits::{Arg, Register};

// taken from criterion
fn cargo_target_directory() -> Option<PathBuf> {
    #[derive(Deserialize)]
    #[allow(dead_code)]
    struct Metadata {
        target_directory: PathBuf,
    }

    std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            let output = Command::new(std::env::var_os("CARGO")?)
                .args(["metadata", "--format-version", "1"])
                .output()
                .ok()?;
            let metadata: Metadata = serde_json::from_slice(&output.stdout).ok()?;
            Some(metadata.target_directory)
        })
}

// taken from the stdlib
pub const fn isqrt(this: u128) -> u128 {
    if this < 2 {
        return this;
    }

    // The algorithm is based on the one presented in
    // <https://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Binary_numeral_system_(base_2)>
    // which cites as source the following C code:
    // <https://web.archive.org/web/20120306040058/http://medialab.freaknet.org/martin/src/sqrt/sqrt.c>.

    let mut op = this;
    let mut res = 0;
    let mut one = 1 << (this.ilog2() & !1);

    while one != 0 {
        if op >= res + one {
            op -= res + one;
            res = (res >> 1) + one;
        } else {
            res >>= 1;
        }
        one >>= 2;
    }

    res
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Nil;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Cons<Head, Tail> {
    pub head: Head,
    pub tail: Tail,
}

pub mod traits {
    use super::*;

    pub trait DebugList {
        fn push_debug(this: &Self, debug: &mut fmt::DebugList<'_, '_>);
    }

    pub trait Arg: Any + fmt::Debug + DynClone {}

    pub trait Register<T> {
        fn push_name(this: &Self, names: &mut Vec<String>);
        fn push_self(this: Self, boxed: &mut Vec<Box<dyn FnMut(Bencher, Box<dyn Arg>)>>);
    }

    impl<T: Any + fmt::Debug + DynClone> Arg for T {}
}

impl traits::DebugList for Nil {
    fn push_debug(this: &Self, debug: &mut fmt::DebugList<'_, '_>) {
        _ = this;
        _ = debug;
    }
}
impl<Head: fmt::Debug, Tail: traits::DebugList> traits::DebugList for Cons<Head, Tail> {
    fn push_debug(this: &Self, debug: &mut fmt::DebugList<'_, '_>) {
        debug.entry(&this.head);
        Tail::push_debug(&this.tail, debug)
    }
}

impl fmt::Debug for Nil {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().finish()
    }
}
impl<Head: fmt::Debug, Tail: traits::DebugList> fmt::Debug for Cons<Head, Tail> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_list();
        <Cons<Head, Tail> as traits::DebugList>::push_debug(self, &mut debug);
        debug.finish()
    }
}

#[macro_export]
macro_rules! unlist {
    () => {
        $crate::Nil
    };
    ($head: pat $(, $tail: pat)* $(,)?) => {
        $crate::Cons {
            head: $head,
            tail: $crate::unlist!($($tail,)*),
        }
    };
}

#[macro_export]
macro_rules! list {
    () => {
        { $crate::Nil }
    };
    ($head: expr $(, $tail: expr)* $(,)?) => {
        $crate::Cons {
            head: $head,
            tail: $crate::list!($($tail,)*),
        }
    };
}

#[macro_export]
macro_rules! List {
    () => {
        $crate::Nil
    };
    ($head: ty $(, $tail: ty)* $(,)?) => {
        $crate::Cons::<$head, $crate::List!($($tail,)*)>
    };
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Picoseconds(pub i128);

impl std::ops::Add for Picoseconds {
    type Output = Picoseconds;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}
impl std::ops::AddAssign for Picoseconds {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0
    }
}

impl std::ops::Sub for Picoseconds {
    type Output = Picoseconds;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}
impl std::ops::SubAssign for Picoseconds {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0
    }
}

impl std::ops::Mul<i128> for Picoseconds {
    type Output = Picoseconds;

    #[inline]
    fn mul(self, rhs: i128) -> Self::Output {
        Self(self.0 * rhs)
    }
}
impl std::ops::Div<i128> for Picoseconds {
    type Output = Picoseconds;

    #[inline]
    fn div(self, rhs: i128) -> Self::Output {
        Self(self.0 / rhs)
    }
}
impl std::ops::Mul<Picoseconds> for i128 {
    type Output = Picoseconds;

    #[inline]
    fn mul(self, rhs: Picoseconds) -> Self::Output {
        Picoseconds(self * rhs.0)
    }
}
impl std::ops::MulAssign<i128> for Picoseconds {
    #[inline]
    fn mul_assign(&mut self, rhs: i128) {
        self.0 *= rhs
    }
}
impl std::ops::DivAssign<i128> for Picoseconds {
    #[inline]
    fn div_assign(&mut self, rhs: i128) {
        self.0 /= rhs
    }
}

impl fmt::Debug for Picoseconds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pico = self.0 as f64;
        let nano = pico / 1e3;
        let micro = pico / 1e6;
        let milli = pico / 1e9;
        let sec = pico / 1e12;
        if pico < 1e3 {
            write!(f, "{pico:6.2} ps")
        } else if nano < 1e3 {
            write!(f, "{nano:6.2} ns")
        } else if micro < 1e3 {
            write!(f, "{micro:6.2} µs")
        } else if milli < 1e3 {
            write!(f, "{milli:6.2} ms")
        } else if sec < 1e3 {
            write!(f, "{sec:6.2}  s")
        } else {
            write!(f, "{sec:6.1e}  s")
        }
    }
}

struct BenchCtx {
    timings: Vec<Picoseconds>,
}

pub struct Bencher<'a> {
    ctx: &'a mut BenchCtx,
    config: &'a BenchConfig,
}

#[inline]
fn time(f: &mut impl FnMut(), iters_per_sample: u64) -> Duration {
    let now = std::time::Instant::now();
    for _ in 0..iters_per_sample {
        f();
    }
    now.elapsed()
}

impl Bencher<'_> {
    pub fn bench<R>(self, f: impl FnMut() -> R) {
        let mut f = f;
        let f = &mut || {
            std::hint::black_box(f());
        };

        self.ctx.timings.clear();
        self.ctx
            .timings
            .reserve_exact(self.config.sample_count.0.try_into().unwrap());

        match self.config.iter_count {
            ItersPerSample::Auto => {
                let mut iters_per_sample = 1;
                let mut done = false;

                loop {
                    let time = time(f, iters_per_sample);
                    if time > self.config.max_time.0 {
                        self.ctx.timings.push(Picoseconds(
                            (time.as_nanos() as i128 * 1000) / iters_per_sample as i128,
                        ));
                        done = true;
                        break;
                    }

                    if time > Duration::from_micros(10) {
                        break;
                    }
                    iters_per_sample *= 2;
                }

                if !done {
                    self.do_bench(f, iters_per_sample);
                }
            }
            ItersPerSample::Manual(iters_per_sample) => {
                self.do_bench(f, iters_per_sample);
            }
        }
    }

    fn do_bench(self, f: &mut impl FnMut(), iters_per_sample: u64) {
        let mut total_time = Duration::ZERO;
        for _ in 0..self.config.sample_count.0 {
            let time = time(f, iters_per_sample);
            self.ctx.timings.push(Picoseconds(
                (time.as_nanos() as i128 * 1000) / iters_per_sample as i128,
            ));
            total_time += time;

            if total_time > self.config.max_time.0 {
                break;
            }
        }
        while total_time < self.config.min_time.0 {
            let time = time(f, iters_per_sample);
            self.ctx.timings.push(Picoseconds(
                (time.as_nanos() as i128 * 1000) / iters_per_sample as i128,
            ));
            total_time += time;

            if total_time > self.config.max_time.0 {
                break;
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SampleCount(pub u64);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PlotSize {
    pub x: u32,
    pub y: u32,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum PlotAxis {
    #[default]
    Linear,
    SemiLogX,
    SemiLogY,
    LogLog,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ItersPerSample {
    Auto,
    Manual(u64),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MinTime(pub Duration);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MaxTime(pub Duration);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TerminalOutput {
    Quiet,
    Verbose,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Split {
    None,
    ByGroup,
    ByArg,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlotName(pub String);

impl Default for SampleCount {
    fn default() -> Self {
        Self(100)
    }
}

impl Default for ItersPerSample {
    fn default() -> Self {
        Self::Auto
    }
}

impl Default for MinTime {
    fn default() -> Self {
        Self(Duration::from_millis(100))
    }
}
impl Default for MaxTime {
    fn default() -> Self {
        Self(Duration::from_secs(5))
    }
}

impl Default for TerminalOutput {
    fn default() -> Self {
        Self::Verbose
    }
}

impl Default for Split {
    fn default() -> Self {
        Self::ByGroup
    }
}

impl Default for PlotSize {
    fn default() -> Self {
        Self { x: 640, y: 400 }
    }
}

impl Default for PlotName {
    fn default() -> Self {
        Self("plot".to_string())
    }
}

#[derive(Debug, Clone, Default)]
pub struct BenchConfig {
    pub sample_count: SampleCount,
    pub iter_count: ItersPerSample,
    pub min_time: MinTime,
    pub max_time: MaxTime,
    pub verbose: TerminalOutput,
    pub split: Split,
    pub plot_size: PlotSize,
    pub plot_axis: PlotAxis,
    pub plot_name: PlotName,
    pub fn_regex: Option<Regex>,
    pub arg_regex: Option<Regex>,
}

impl BenchConfig {
    pub fn new() -> Self {
        Default::default()
    }
}

pub struct Bench {
    pub config: BenchConfig,
    pub fns: Vec<(
        Vec<(String, Box<dyn FnMut(Bencher, Box<dyn Arg>)>)>,
        Vec<Box<dyn Arg>>,
    )>,
}

impl<T> traits::Register<T> for Nil {
    fn push_name(_: &Self, _: &mut Vec<String>) {}
    fn push_self(_: Self, _: &mut Vec<Box<dyn FnMut(Bencher, Box<dyn Arg>)>>) {}
}

impl<T: Arg, Head: 'static + FnMut(Bencher, T), Tail: traits::Register<T>> traits::Register<T>
    for Cons<Head, Tail>
{
    fn push_name(this: &Self, names: &mut Vec<String>) {
        names.push(std::any::type_name_of_val(&this.head).to_string());
        Tail::push_name(&this.tail, names);
    }
    fn push_self(this: Self, boxed: &mut Vec<Box<dyn FnMut(Bencher, Box<dyn Arg>)>>) {
        let mut f = this.head;
        boxed.push(Box::new(move |bencher, arg| {
            assert!((*arg).type_id() == TypeId::of::<T>());
            let arg: Box<T> = unsafe { Box::from_raw(Box::into_raw(arg) as *mut T) };
            f(bencher, *arg);
        }));
        Tail::push_self(this.tail, boxed);
    }
}

impl Bench {
    pub fn new(config: BenchConfig) -> Self {
        Self {
            config,
            fns: Vec::new(),
        }
    }

    pub fn register_many_with_names<T: Arg, F: traits::Register<T>>(
        &mut self,
        names: Vec<String>,
        f: F,
        args: Vec<T>,
    ) {
        let mut boxed = Vec::new();
        traits::Register::push_self(f, &mut boxed);

        self.fns.push((
            std::iter::zip(names, boxed).collect(),
            args.into_iter()
                .map(|arg| Box::new(arg) as Box<dyn Arg>)
                .collect(),
        ));
    }

    pub fn register_many<T: Arg>(&mut self, f: impl Register<T>, args: Vec<T>) {
        let mut names = Vec::new();
        Register::push_name(&f, &mut names);
        self.register_many_with_names(names, f, args);
    }

    pub fn register<T: Arg>(&mut self, f: impl 'static + FnMut(Bencher, T), args: Vec<T>) {
        self.register_many(list![f], args)
    }

    pub fn register_with_name<T: Arg>(
        &mut self,
        name: impl AsRef<str>,
        f: impl 'static + FnMut(Bencher, T),
        args: Vec<T>,
    ) {
        self.register_many_with_names(vec![name.as_ref().to_string()], list![f], args)
    }

    pub fn run(&mut self) {
        let config = &self.config;
        let plot_name = &config.plot_name.0;

        let mut max_name_len = 14;
        let mut max_arg_len = 4;

        for (fns, args) in &self.fns {
            for (name, _) in fns {
                if config
                    .fn_regex
                    .as_ref()
                    .is_some_and(|regex| !regex.is_match(name))
                {
                    continue;
                }
                max_name_len = Ord::max(max_name_len, name.len());
            }
            for arg in args {
                let arg = &*format!("{arg:?}");
                if config
                    .arg_regex
                    .as_ref()
                    .is_some_and(|regex| !regex.is_match(arg))
                {
                    continue;
                }
                max_arg_len = Ord::max(max_arg_len, arg.len());
            }
        }

        max_name_len += 1;
        max_arg_len += 1;

        let verbose = config.verbose == TerminalOutput::Verbose;
        if verbose {
            println!(
                "╭─{:─<max_name_len$}┬{:─>max_arg_len$}─┬─{:─<9}─┬─{:─<9}─┬─{:─<9}─┬─{:─<9}─╮",
                "", "", "", "", "", "",
            );
            println!(
                "│ {:<max_name_len$}│{:>max_arg_len$} │ {:>9} │ {:>9} │ {:>9} │ {:>9} │",
                "benchmark", "args", "fastest", "median", "mean", "stddev",
            );
        }

        let mut plot_id = 0;

        for (fns, args) in &mut self.fns {
            let target = cargo_target_directory()
                .unwrap()
                .join(format!("{plot_name}_{plot_id}.svg"));
            let mut max_arg = 0usize;
            let mut min_arg = usize::MAX;
            let mut max_time = Picoseconds(0);
            let mut lines = vec![(RGBAColor(0, 0, 0, 1.0), String::new(), Vec::new()); fns.len()];

            for arg in &**args {
                if (**arg).type_id() == TypeId::of::<PlotArg>() {
                    let arg = unsafe { &*(&**arg as *const dyn Arg as *const PlotArg) };
                    max_arg = Ord::max(arg.0, max_arg);
                    min_arg = Ord::min(arg.0, min_arg);
                }
            }

            if verbose && matches!(config.split, Split::ByGroup | Split::ByArg) {
                println!(
                    "├─{:─<max_name_len$}┼{:─>max_arg_len$}─┼─{:─<9}─┼─{:─<9}─┼─{:─<9}─┼─{:─<9}─┤",
                    "", "", "", "", "", "",
                );
            }
            let args = &**args;
            for arg in args {
                let arg_str = &*format!("{arg:?}");
                if config
                    .arg_regex
                    .as_ref()
                    .is_some_and(|regex| !regex.is_match(arg_str))
                {
                    continue;
                }

                if verbose && config.split == Split::ByArg {
                    println!(
                        "├─{:─<max_name_len$}┼{:─>max_arg_len$}─┼─{:─<9}─┼─{:─<9}─┼─{:─<9}─┼─{:─<9}─┤",
                        "", "", "", "", "", "",
                    );
                }

                let fn_count = fns.len();
                for (idx, (name, f)) in fns.iter_mut().enumerate() {
                    let name = &**name;
                    if config
                        .fn_regex
                        .as_ref()
                        .is_some_and(|regex| !regex.is_match(name))
                    {
                        continue;
                    }

                    let f = &mut **f;
                    let ctx = &mut BenchCtx {
                        timings: Vec::new(),
                    };
                    f(Bencher { ctx, config }, dyn_clone::clone_box(&**arg));
                    ctx.timings.sort_unstable();
                    let count = ctx.timings.len();

                    if count > 0 {
                        let sum = ctx.timings.iter().map(|x| x.0).sum::<i128>();
                        let mean = Picoseconds(sum / count as i128);

                        let variance = if count == 1 {
                            0
                        } else {
                            ctx.timings
                                .iter()
                                .map(|x| {
                                    let diff = x.0 - mean.0;
                                    diff * diff
                                })
                                .sum::<i128>()
                                / (count as i128 - 1)
                        };
                        let stddev = Picoseconds(isqrt(variance as u128) as i128);

                        let fastest = ctx.timings[0];
                        let median = ctx.timings[count / 2];

                        if max_arg > 0 {
                            assert!((**arg).type_id() == TypeId::of::<PlotArg>());
                            let arg = unsafe { &*(&**arg as *const dyn Arg as *const PlotArg) };
                            max_time = Ord::max(max_time, fastest);
                            lines[idx].0 = ViridisRGBA::get_color(if fn_count == 0 {
                                0.5
                            } else {
                                idx as f32 / (fn_count - 1) as f32
                            });
                            if lines[idx].1.is_empty() {
                                lines[idx].1 = name.to_string();
                            }
                            lines[idx].2.push((arg.0, fastest));
                        }

                        if verbose {
                            println!(
                                "│ {name:<max_name_len$}│{arg_str:>max_arg_len$} │ {fastest:?} │ {median:?} │ {mean:?} │ {stddev:?} │"
                            );
                        }
                    }
                }
            }

            if max_arg > 0 {
                let root = SVGBackend::new(&target, (config.plot_size.x, config.plot_size.y))
                    .into_drawing_area();
                root.fill(&GREY_300).unwrap();
                let mut builder = ChartBuilder::on(&root);
                builder
                    .margin(30)
                    .x_label_area_size(30)
                    .y_label_area_size(30);

                let xrange = min_arg as f32..max_arg as f32;
                let yrange = 0.0f32..max_time.0 as f32 / 1e12;

                match config.plot_axis {
                    PlotAxis::Linear => do_plot(builder, xrange, yrange, &mut plot_id, lines),
                    PlotAxis::SemiLogX => {
                        do_plot(builder, xrange.log_scale(), yrange, &mut plot_id, lines)
                    }
                    PlotAxis::SemiLogY => {
                        do_plot(builder, xrange, yrange.log_scale(), &mut plot_id, lines)
                    }
                    PlotAxis::LogLog => do_plot(
                        builder,
                        xrange.log_scale(),
                        yrange.log_scale(),
                        &mut plot_id,
                        lines,
                    ),
                }

                root.present().unwrap();
            }
        }

        if verbose {
            println!(
                "╰─{:─<max_name_len$}┴{:─>max_arg_len$}─┴─{:─<9}─┴─{:─<9}─┴─{:─<9}─┴─{:─<9}─╯",
                "", "", "", "", "", "",
            );
        }
    }
}

fn do_plot<'a, X: AsRangedCoord, Y: AsRangedCoord>(
    mut builder: ChartBuilder<'_, '_, SVGBackend<'a>>,
    xrange: X,
    yrange: Y,
    plot_id: &mut i32,
    lines: Vec<(RGBAColor, String, Vec<(usize, Picoseconds)>)>,
) where
    X::CoordDescType: ValueFormatter<X::Value>,
    Y::CoordDescType: ValueFormatter<Y::Value>,
    for<'b> &'b DynElement<'static, SVGBackend<'a>, (f32, f32)>:
        PointCollection<'b, (X::Value, Y::Value)>,
{
    let mut chart = builder.build_cartesian_2d(xrange, yrange).unwrap();
    chart.configure_mesh().max_light_lines(2).draw().unwrap();
    *plot_id += 1;

    for (color, name, line) in &lines {
        let style = ShapeStyle {
            color: *color,
            filled: false,
            stroke_width: 3,
        };
        chart
            .draw_series(LineSeries::new(
                line.iter()
                    .map(|&(n, time)| (n as f32, time.0 as f32 / 1e12)),
                style,
            ))
            .unwrap()
            .label(name)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));
    }

    chart
        .configure_series_labels()
        .background_style(&GREY_A100.mix(0.8))
        .border_style(&full_palette::BLACK)
        .draw()
        .unwrap();
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PlotArg(pub usize);

impl fmt::Debug for PlotArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

pub mod prelude {
    pub use super::{
        list, unlist, Bench, BenchConfig, Bencher, Cons, ItersPerSample, List, MaxTime, MinTime,
        PlotArg, PlotAxis, PlotName, PlotSize, SampleCount, Split, TerminalOutput,
    };
    pub use regex::Regex;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list() {
        let unlist![_, _, _]: List![i32, u32, usize] = list![1, 2, 3];
        println!("{:?}", list![1, 3, vec![1.0]]);
    }

    fn naive(bencher: Bencher, PlotArg(n): PlotArg) {
        let bytes = vec![1u8; n];
        bencher.bench(|| {
            let mut count = 0u64;
            for byte in &bytes {
                count += byte.count_ones() as u64;
            }
            count
        })
    }

    fn popcnt(bencher: Bencher, PlotArg(n): PlotArg) {
        let bytes = vec![1u8; n];
        bencher.bench(|| popcnt::count_ones(&bytes))
    }

    #[test]
    fn test_bench() {
        let mut bench = Bench::new(BenchConfig {
            split: Split::ByArg,
            plot_axis: PlotAxis::LogLog,
            min_time: MinTime(Duration::from_millis(100)),
            max_time: MaxTime(Duration::from_millis(100)),
            ..Default::default()
        });
        println!();
        bench.register_many(
            list![naive, popcnt],
            (0..20).map(|i| 1 << i).map(PlotArg).collect(),
        );
        bench.run();
    }
}
