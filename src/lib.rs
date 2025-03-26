//! `diol` is a benchmarking library for rust.
//!
//! # getting started
//! add the following to your `Cargo.toml`.
//! ```notcode
//! [dev-dependencies]
//! diol = "0.10"
//!
//! [[bench]]
//! name = "my_benchmark"
//! harness = false
//! ```
//! then in `benches/my_benchmark.rs`.
//!
//! ```rust
//! use diol::prelude::*;
//!
//! fn main() -> eyre::Result<()> {
//!     let mut bench = Bench::new(BenchConfig::from_args()?);
//!     bench.register(slice_times_two, [4, 8, 16, 128, 1024]);
//!     bench.run()?;
//!     Ok(())
//! }
//!
//! fn slice_times_two(bencher: Bencher, len: usize) {
//!     let mut v = vec![0.0_f64; len];
//!     bencher.bench(|| {
//!         for x in &mut v {
//!             *x *= 2.0;
//!         }
//!         black_box(&mut v);
//!     });
//! }
//! ```
//!
//! run the benchmark with `cargo bench`, or `cargo bench --bench my_benchmark` if you have multiple
//! benchmarks you can also pass in benchmark options using `cargo bench --bench my_benchmark --
//! [OPTIONS...]`
//!
//! ```notcode
//! ╭─────────────────┬──────┬───────────┬───────────┬───────────┬───────────╮
//! │ benchmark       │ args │   fastest │    median │      mean │    stddev │
//! ├─────────────────┼──────┼───────────┼───────────┼───────────┼───────────┤
//! │ slice_times_two │    4 │  29.61 ns │  34.38 ns │  34.83 ns │   1.62 ns │
//! ├─────────────────┼──────┼───────────┼───────────┼───────────┼───────────┤
//! │ slice_times_two │    8 │  44.17 ns │  53.04 ns │  53.32 ns │   3.25 ns │
//! ├─────────────────┼──────┼───────────┼───────────┼───────────┼───────────┤
//! │ slice_times_two │   16 │  93.66 ns │ 107.91 ns │ 108.13 ns │   4.11 ns │
//! ├─────────────────┼──────┼───────────┼───────────┼───────────┼───────────┤
//! │ slice_times_two │  128 │ 489.97 ns │ 583.59 ns │ 585.28 ns │  33.15 ns │
//! ├─────────────────┼──────┼───────────┼───────────┼───────────┼───────────┤
//! │ slice_times_two │ 1024 │   3.77 µs │   4.51 µs │   4.53 µs │ 173.44 ns │
//! ╰─────────────────┴──────┴───────────┴───────────┴───────────┴───────────╯
//! ```
use clap::Parser;
use dyn_clone::DynClone;
use equator::assert;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    any::{Any, TypeId},
    cell::RefCell,
    fmt,
    io::Write,
    path::PathBuf,
    process::Command,
    time::Duration,
};

use config::*;
use result::*;
use traits::{Arg, Register, RegisterMany};
use variadics::{Cons, Nil};

#[cfg(feature = "plot")]
mod typst_imp;

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
const fn isqrt(this: u128) -> u128 {
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

#[derive(Clone)]
struct TimeMetric;

impl traits::PlotMetric for TimeMetric {
    fn name(&self) -> &'static str {
        "time (μs)"
    }
    fn compute(&self, _: PlotArg, time: Picoseconds) -> f64 {
        time.0 as f64 / 1e6
    }

    fn monotonicity(&self) -> traits::Monotonicity {
        traits::Monotonicity::LowerIsBetter
    }
}

trait DebugList {
    fn push_debug(this: &Self, debug: &mut fmt::DebugList<'_, '_>);
}

/// helper traits and types
pub mod traits {
    use super::*;

    /// boxed `Register` trait object.
    pub struct DynRegister<T>(pub Box<dyn Register<T>>);

    /// type that can be used as a benchmark argument.
    pub trait Arg: Any + fmt::Debug + DynClone {}

    /// type that can be registered as a benchmark function.
    pub trait Register<T>: 'static {
        fn get_name(&self) -> String;
        fn call_mut(&mut self, bencher: Bencher, arg: T);
    }

    /// [`Register`] extension trait.
    pub trait RegisterExt<T>: Register<T> + Sized {
        fn boxed(self) -> DynRegister<T> {
            DynRegister(Box::new(self))
        }
        fn with_name(self, name: impl AsRef<str>) -> impl Register<T> {
            fn implementation<T, F: Register<T>>(this: F, name: &str) -> impl Register<T> {
                struct Named<F>(String, F);
                impl<T, F: Register<T>> Register<T> for Named<F> {
                    fn get_name(&self) -> String {
                        self.0.clone()
                    }

                    fn call_mut(&mut self, bencher: Bencher, arg: T) {
                        self.1.call_mut(bencher, arg)
                    }
                }

                Named(name.to_string(), this)
            }

            implementation(self, name.as_ref())
        }
    }

    impl<T, F: Register<T>> RegisterExt<T> for F {}
    impl<T: Arg> Register<T> for DynRegister<T> {
        fn get_name(&self) -> String {
            (*self.0).get_name()
        }

        fn call_mut(&mut self, bencher: Bencher, arg: T) {
            (*self.0).call_mut(bencher, arg)
        }
    }

    impl<T, F: FnMut(Bencher, T) + 'static> Register<T> for F {
        fn get_name(&self) -> String {
            let name = std::any::type_name::<Self>();
            if let Ok(mut ty) = syn::parse_str::<syn::Type>(name) {
                minify_ty(&mut ty);
                let file = syn::parse2::<syn::File>(quote::quote! { type X = [#ty]; }).unwrap();
                let file = prettyplease::unparse(&file);
                let mut file = &*file;
                file = &file[file.find('[').unwrap() + 1..];
                file = &file[..file.rfind(']').unwrap()];
                file.to_string()
            } else {
                name.to_string()
            }
        }

        fn call_mut(&mut self, bencher: Bencher, arg: T) {
            (*self)(bencher, arg)
        }
    }

    /// variadic tuple of types that can be registered as benchmark functions.
    pub trait RegisterMany<T> {
        fn push_name(this: &Self, names: &mut Vec<String>);
        fn push_self(this: Self, boxed: &mut Vec<Box<dyn Register<Box<dyn Arg>>>>);
    }

    /// whether the plot metric is better when higher, lower, or not monotonic
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    pub enum Monotonicity {
        None,
        HigherIsBetter,
        LowerIsBetter,
    }

    /// type that can be used as a metric for plots.
    pub trait PlotMetric: DynClone + Any {
        fn compute(&self, arg: PlotArg, time: Picoseconds) -> f64;
        fn monotonicity(&self) -> Monotonicity;
        fn name(&self) -> &str {
            std::any::type_name::<Self>().split("::").last().unwrap()
        }
    }

    impl<T: 'static + DynClone + Fn(PlotArg, Picoseconds) -> f64> PlotMetric for T {
        fn monotonicity(&self) -> Monotonicity {
            Monotonicity::None
        }
        fn compute(&self, arg: PlotArg, time: Picoseconds) -> f64 {
            (*self)(arg, time)
        }
    }

    impl<T: Any + fmt::Debug + DynClone> Arg for T {}
}

impl DebugList for Nil {
    fn push_debug(this: &Self, debug: &mut fmt::DebugList<'_, '_>) {
        _ = this;
        _ = debug;
    }
}
impl<Head: fmt::Debug, Tail: DebugList> DebugList for Cons<Head, Tail> {
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
impl<Head: fmt::Debug, Tail: DebugList> fmt::Debug for Cons<Head, Tail> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_list();
        <Cons<Head, Tail> as DebugList>::push_debug(self, &mut debug);
        debug.finish()
    }
}

#[macro_export]
#[doc(hidden)]
macro_rules! __list_impl {
    (@ __impl @ () @ ()) => {
        $crate::variadics::Nil
    };

    (@ __impl @ ($($parsed:tt)+) @ ()) => {
        $crate::variadics::Cons {
            head: $($parsed)+,
            tail: $crate::variadics::Nil,
        }
    };

    (@ __impl @ ($($parsed:tt)+) @ (, $($unparsed:tt)*)) => {
        $crate::variadics::Cons {
            head: $($parsed)+,
            tail: $crate::__list_impl![@ __impl @ () @ ($($unparsed)*)],
        }
    };

    (@ __impl @ ($($parsed:tt)*) @ ($unparsed_head: tt $($unparsed_rest:tt)*)) => {
        $crate::__list_impl![@ __impl @ ($($parsed)* $unparsed_head) @ ($($unparsed_rest)*)]
    };
}

/// create a variadic tuple containing the given values.
#[macro_export]
macro_rules! list {
    () => {
        $crate::variadics::Nil
    };
    ($head: expr $(, $tail: expr)* $(,)?) => {
        $crate::variadics::Cons {
            head: $head,
            tail: $crate::list!($($tail,)*)
        }
    };
}

/// destructure a variadic tuple containing the given values.
#[macro_export]
macro_rules! unlist {
    () => {
        $crate::variadics::Nil
    };
    ($head: pat $(, $tail: pat)* $(,)?) => {
        $crate::variadics::Cons {
            head: $head,
            tail: $crate::unlist!($($tail,)*)
        }
    };
}

/// type of a variadic tuple containing the given types.
#[macro_export]
macro_rules! List {
    () => {
        $crate::variadics::Nil
    };
    ($head: ty $(, $tail: ty)* $(,)?) => {
        $crate::variadics::Cons::<$head, $crate::List!($($tail,)*)>
    };
}

/// extra-precise duration type for short-ish durations.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub struct Picoseconds(pub i128);

impl Picoseconds {
    pub fn to_secs(self) -> f64 {
        self.0 as f64 / 1e12
    }
}

impl std::iter::Sum for Picoseconds {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self(iter.map(|x| x.0).sum())
    }
}

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
        if self.0 == 0 {
            write!(f, "{: ^9}", "-")
        } else if pico < 1e3 {
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

/// bench loop runner.
pub struct Bencher<'a> {
    ctx: &'a mut BenchCtx,
    config: &'a BenchConfig,
}

#[inline]
fn measure_time(f: &mut impl FnMut(), iters_per_sample: u64) -> Duration {
    let now = std::time::Instant::now();
    for _ in 0..iters_per_sample {
        f();
    }
    now.elapsed()
}

impl Bencher<'_> {
    pub fn skip(self) {
        self.ctx.timings.clear();
    }

    /// run the function in a loop and measure the timings.
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
                    let time = measure_time(f, iters_per_sample);
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
            let time = measure_time(f, iters_per_sample);
            self.ctx.timings.push(Picoseconds(
                (time.as_nanos() as i128 * 1000) / iters_per_sample as i128,
            ));
            total_time += time;

            if total_time > self.config.max_time.0 {
                break;
            }
        }
        while total_time < self.config.min_time.0 {
            let time = measure_time(f, iters_per_sample);
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

type BencherGroup = (
    Vec<(String, Box<dyn Register<Box<dyn Arg>>>)>,
    (TypeId, Vec<Box<dyn Arg>>),
);

/// main benchmark entry point, used to register functions and arguments, then run benchmarks.
pub struct Bench {
    pub config: BenchConfig,
    pub groups: RefCell<Vec<BencherGroup>>,
}

impl<T> traits::RegisterMany<T> for Nil {
    fn push_name(_: &Self, _: &mut Vec<String>) {}
    fn push_self(_: Self, _: &mut Vec<Box<dyn Register<Box<dyn Arg>>>>) {}
}

fn minify_path_segment(segment: &mut syn::PathSegment) {
    match &mut segment.arguments {
        syn::PathArguments::None => {}
        syn::PathArguments::AngleBracketed(t) => {
            for arg in &mut t.args {
                match arg {
                    syn::GenericArgument::Lifetime(_) => {}
                    syn::GenericArgument::Type(t) => minify_ty(t),
                    syn::GenericArgument::Const(_) => {}
                    syn::GenericArgument::AssocType(t) => {
                        minify_ty(&mut t.ty);
                    }
                    syn::GenericArgument::AssocConst(_) => {}
                    syn::GenericArgument::Constraint(_) => {}
                    _ => {}
                }
            }
        }
        syn::PathArguments::Parenthesized(t) => {
            for t in &mut t.inputs {
                minify_ty(t);
            }
            match &mut t.output {
                syn::ReturnType::Default => {}
                syn::ReturnType::Type(_, t) => {
                    minify_ty(t);
                }
            }
        }
    }
}

fn minify_bound(bound: &mut syn::TypeParamBound) {
    if let syn::TypeParamBound::Trait(t) = bound {
        t.path.leading_colon = None;
        if let Some(last) = t.path.segments.pop() {
            let mut last = last.into_value();
            minify_path_segment(&mut last);
            t.path.segments.clear();
            t.path.segments.push_value(last);
        }
    }
}

fn minify_ty(ty: &mut syn::Type) {
    use syn::Type;
    match ty {
        Type::Array(t) => minify_ty(&mut t.elem),
        Type::BareFn(t) => {
            for arg in &mut t.inputs {
                minify_ty(&mut arg.ty);
            }
            if let syn::ReturnType::Type(_, ty) = &mut t.output {
                minify_ty(ty);
            }
        }
        Type::Group(t) => minify_ty(&mut t.elem),
        Type::ImplTrait(t) => {
            for bound in &mut t.bounds {
                minify_bound(bound);
            }
        }

        Type::Infer(_) => {}
        Type::Macro(_) => {}
        Type::Never(_) => {}
        Type::Paren(_) => {}
        Type::Path(t) => {
            if let Some(last) = t.path.segments.pop() {
                let mut last = last.into_value();
                minify_path_segment(&mut last);
                t.path.segments.clear();
                t.path.segments.push_value(last);
            }
        }
        Type::Ptr(t) => minify_ty(&mut t.elem),
        Type::Reference(t) => minify_ty(&mut t.elem),
        Type::Slice(t) => minify_ty(&mut t.elem),
        Type::TraitObject(t) => {
            for bound in &mut t.bounds {
                minify_bound(bound);
            }
        }
        Type::Tuple(t) => {
            for t in &mut t.elems {
                minify_ty(t);
            }
        }
        Type::Verbatim(_) => {}
        _ => {}
    };
}

impl<T: Arg, Head: Register<T>, Tail: traits::RegisterMany<T>> traits::RegisterMany<T>
    for Cons<Head, Tail>
{
    fn push_name(this: &Self, names: &mut Vec<String>) {
        names.push(Head::get_name(&this.head));
        Tail::push_name(&this.tail, names);
    }
    fn push_self(this: Self, boxed: &mut Vec<Box<dyn Register<Box<dyn Arg>>>>) {
        let mut f = this.head;
        boxed.push(Box::new(move |bencher: Bencher<'_>, arg: Box<dyn Arg>| {
            assert!((*arg).type_id() == TypeId::of::<T>());
            let arg: Box<T> = unsafe { Box::from_raw(Box::into_raw(arg) as *mut T) };
            f.call_mut(bencher, *arg);
        }));
        Tail::push_self(this.tail, boxed);
    }
}

impl AsRef<BenchConfig> for BenchConfig {
    fn as_ref(&self) -> &BenchConfig {
        self
    }
}

impl Bench {
    /// create a bench object from the given configuration.
    pub fn new(config: impl AsRef<BenchConfig>) -> Self {
        Self {
            config: config.as_ref().clone(),
            groups: RefCell::new(Vec::new()),
        }
    }

    #[doc(hidden)]
    pub unsafe fn register_many_dyn(
        &self,
        names: Vec<String>,
        boxed: Vec<Box<dyn Register<Box<dyn Arg>>>>,
        type_id: TypeId,
        args: Vec<Box<dyn Arg>>,
    ) {
        self.groups
            .borrow_mut()
            .push((std::iter::zip(names, boxed).collect(), (type_id, args)))
    }

    fn register_many_with_names<T: Arg, F: traits::RegisterMany<T>>(
        &self,
        names: Vec<String>,
        f: F,
        args: impl IntoIterator<Item = T>,
    ) {
        let mut boxed = Vec::new();
        traits::RegisterMany::push_self(f, &mut boxed);

        self.groups.borrow_mut().push((
            std::iter::zip(names, boxed).collect(),
            (
                TypeId::of::<T>(),
                args.into_iter()
                    .map(|arg| Box::new(arg) as Box<dyn Arg>)
                    .collect(),
            ),
        ));
    }

    /// register multiple functions that should be compared against each other during benchmarking,
    /// all taking the same arguments.
    pub fn register_many<T: Arg>(
        &self,
        f: impl RegisterMany<T>,
        args: impl IntoIterator<Item = T>,
    ) {
        let mut names = Vec::new();
        RegisterMany::push_name(&f, &mut names);
        self.register_many_with_names(names, f, args);
    }

    /// register a function for benchmarking, and the arguments that should be passed to it.
    pub fn register<T: Arg>(&mut self, f: impl Register<T>, args: impl IntoIterator<Item = T>) {
        self.register_many(list![f], args)
    }

    /// run the benchmark, and write the results to stdout, and optionally to a file, depending on
    /// the configuration options.
    pub fn run(&self) -> eyre::Result<BenchResult> {
        let config = &self.config;
        let mut result = BenchResult { groups: Vec::new() };

        let verbose = config.verbose == StdoutPrint::Verbose;

        let mut plot_id = 0;

        for (group, (type_id, args)) in &mut *self.groups.borrow_mut() {
            let mut nargs = 0;
            let mut nfuncs = 0;
            let mut max_name_len = 14;
            let mut max_arg_len = 4;

            for (name, _) in &**group {
                if config
                    .func_filter
                    .as_ref()
                    .is_some_and(|regex| !regex.is_match(name))
                {
                    continue;
                }
                nfuncs += 1;
                max_name_len = Ord::max(max_name_len, name.len());
            }
            for arg in &**args {
                let arg = &*format!("{arg:?}");
                if config
                    .arg_filter
                    .as_ref()
                    .is_some_and(|regex| !regex.is_match(arg))
                {
                    continue;
                }
                nargs += 1;
                max_arg_len = Ord::max(max_arg_len, arg.len());
            }

            max_name_len += 1;
            max_arg_len += 1;

            if nargs == 0 || nfuncs == 0 {
                continue;
            }

            let is_plot_arg =
                (*type_id == TypeId::of::<PlotArg>()) || (*type_id == TypeId::of::<usize>());

            let is_not_time_metric =
                (*config.plot_metric.0).type_id() != TypeId::of::<TimeMetric>();
            let metric_name = config.plot_metric.0.name().to_string();
            let metric_mono = config.plot_metric.0.monotonicity();
            let metric_len = Ord::max(9, metric_name.len() + 1);

            if verbose {
                let mut stdout = std::io::stdout();
                if is_plot_arg && is_not_time_metric {
                    writeln!(
                        stdout,
                        "╭─{:─<max_name_len$}┬{:─>max_arg_len$}─┬{:─>metric_len$}─┬─{:─<9}─┬─{:─<9}─┬─{:─<9}─┬─{:─<9}─╮",
                        "", "", "", "", "", "", "",
                    )?;
                    writeln!(
                        stdout,
                        "│ {:<max_name_len$}│{:>max_arg_len$} │{:>metric_len$} │ {:>9} │ {:>9} │ {:>9} │ {:>9} │",
                        "benchmark", "args", metric_name, "fastest", "median", "mean", "stddev",
                    )?;
                } else {
                    writeln!(
                        stdout,
                        "╭─{:─<max_name_len$}┬{:─>max_arg_len$}─┬─{:─<9}─┬─{:─<9}─┬─{:─<9}─┬─{:─<9}─╮",
                        "", "", "", "", "", "",
                    )?;
                    writeln!(
                        stdout,
                        "│ {:<max_name_len$}│{:>max_arg_len$} │ {:>9} │ {:>9} │ {:>9} │ {:>9} │",
                        "benchmark", "args", "fastest", "median", "mean", "stddev",
                    )?;
                }
            }

            let mut group_function_result = Vec::new();
            let mut group_arg_named = Vec::new();
            let mut group_arg_plot = Vec::new();

            let mut max_arg = 0usize;
            let mut min_arg = usize::MAX;
            let mut max_y = f64::NEG_INFINITY;
            let mut min_y = f64::INFINITY;

            for arg in &**args {
                if (**arg).type_id() == TypeId::of::<PlotArg>()
                    || arg.type_id() == TypeId::of::<usize>()
                {
                    let arg = unsafe { &*(&**arg as *const dyn Arg as *const PlotArg) };
                    max_arg = Ord::max(arg.0, max_arg);
                    min_arg = Ord::min(arg.0, min_arg);
                    group_arg_plot.push(*arg);
                } else {
                    let arg = &**arg;
                    group_arg_named.push(format!("{arg:?}"));
                }
            }
            for (name, _) in &**group {
                if config
                    .func_filter
                    .as_ref()
                    .is_some_and(|regex| !regex.is_match(name))
                {
                    continue;
                }

                group_function_result.push(BenchFunctionResult {
                    name: name.to_string(),
                    timings: vec![Vec::new(); args.len()],
                    metric: if is_plot_arg {
                        Some(vec![vec![]; args.len()])
                    } else {
                        None
                    },
                })
            }

            let args = &**args;
            for (arg_idx, arg) in args.iter().enumerate() {
                let arg_str = &*format!("{arg:?}");
                if config
                    .arg_filter
                    .as_ref()
                    .is_some_and(|regex| !regex.is_match(arg_str))
                {
                    continue;
                }

                if verbose {
                    let mut stdout = std::io::stdout();
                    if is_plot_arg && is_not_time_metric {
                        writeln!(
                            stdout,
                            "├─{:─<max_name_len$}┼{:─>max_arg_len$}─┼{:─>metric_len$}─┼─{:─<9}─┼─{:─<9}─┼─{:─<9}─┼─{:─<9}─┤",
                            "", "", "", "", "", "", "",
                        )?;
                    } else {
                        writeln!(
                            stdout,
                            "├─{:─<max_name_len$}┼{:─>max_arg_len$}─┼─{:─<9}─┼─{:─<9}─┼─{:─<9}─┼─{:─<9}─┤",
                            "", "", "", "", "", "",
                        )?;
                    }
                }

                for (idx, (name, f)) in group
                    .iter_mut()
                    .filter(|(name, _)| {
                        !config
                            .func_filter
                            .as_ref()
                            .is_some_and(|regex| !regex.is_match(name))
                    })
                    .enumerate()
                {
                    let name = &**name;

                    let f = &mut **f;
                    let mut ctx = BenchCtx {
                        timings: Vec::new(),
                    };
                    f.call_mut(
                        Bencher {
                            ctx: &mut ctx,
                            config,
                        },
                        dyn_clone::clone_box(&**arg),
                    );
                    ctx.timings.sort_unstable();
                    let mut metric = vec![];
                    let mut metric_mean = 0.0;
                    let count = ctx.timings.len();

                    let (mean, stddev) = result::Stats::from_slice(&ctx.timings).mean_stddev();

                    let fastest = ctx.timings.first().copied().unwrap_or_default();
                    let median = ctx.timings.get(count / 2).copied().unwrap_or_default();

                    if is_plot_arg {
                        let arg = unsafe { &*(&**arg as *const dyn Arg as *const PlotArg) };
                        metric = ctx
                            .timings
                            .iter()
                            .map(|time| config.plot_metric.0.compute(*arg, *time))
                            .collect();
                        metric_mean = result::Stats::from_slice(&metric).mean_stddev().0;

                        max_y = f64::max(max_y, metric_mean);
                        min_y = f64::min(min_y, metric_mean);
                    }

                    if verbose {
                        let mut stdout = std::io::stdout();
                        if is_plot_arg && is_not_time_metric {
                            writeln!(
                                    stdout,
                                    "│ {name:<max_name_len$}│{arg_str:>max_arg_len$} │{metric_mean:>metric_len$.3e} │ {fastest:?} │ {median:?} │ {mean:?} │ {stddev:?} │"
                                )?;
                        } else {
                            writeln!(
                                    stdout,
                                    "│ {name:<max_name_len$}│{arg_str:>max_arg_len$} │ {fastest:?} │ {median:?} │ {mean:?} │ {stddev:?} │"
                                )?;
                        }
                    }

                    group_function_result[idx].timings[arg_idx] = ctx.timings;
                    if let Some(metrics) = &mut group_function_result[idx].metric {
                        metrics[arg_idx] = metric;
                    }
                }
            }

            if !group_arg_plot.is_empty() {
                let group = BenchGroupResult {
                    function: group_function_result,
                    args: BenchArgs::Plot(group_arg_plot),
                    metric_name,
                    metric_mono,
                };

                #[cfg(feature = "plot")]
                if is_plot_arg {
                    if let Some(plot_dir) = &config.plot_dir.0 {
                        let name = &config.plot_name.0;
                        group.plot(&format!("{name}_{plot_id}"), config.plot_axis, plot_dir)?;
                    }
                }

                result.groups.push(group);
            } else {
                let group = BenchGroupResult {
                    function: group_function_result,
                    args: BenchArgs::Named(group_arg_named),
                    metric_name,
                    metric_mono,
                };
                result.groups.push(group);
            }
            if verbose {
                let mut stdout = std::io::stdout();
                if is_plot_arg && is_not_time_metric {
                    writeln!(
                        stdout,
                        "╰─{:─<max_name_len$}┴{:─>max_arg_len$}─┴{:─>metric_len$}─┴─{:─<9}─┴─{:─<9}─┴─{:─<9}─┴─{:─<9}─╯",
                        "", "", "", "", "", "", "",
                    )?;
                } else {
                    writeln!(
                        stdout,
                        "╰─{:─<max_name_len$}┴{:─>max_arg_len$}─┴─{:─<9}─┴─{:─<9}─┴─{:─<9}─┴─{:─<9}─╯",
                        "", "", "", "", "", "",
                    )?;
                }
            }
            plot_id += 1;
        }

        if let Some(path) = &config.output {
            let file = std::fs::File::create(path)?;
            serde_json::ser::to_writer(std::io::BufWriter::new(file), &result)?;
        }

        Ok(result)
    }
}

/// re-exports of useful types and traits.
pub mod prelude {
    pub use crate::{
        config::BenchConfig, list, traits::RegisterExt, Bench, Bencher, List, PlotArg,
    };
    pub use eyre;
    pub use std::hint::black_box;
}
pub extern crate eyre;
pub use eyre::Result;

/// variadic tuple type.
pub mod variadics {
    /// empty tuple.
    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Nil;

    /// non-empty tuple, containing the first element and the rest of the elements as a variadic
    /// tuple.
    #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Cons<Head, Tail> {
        /// first element.
        pub head: Head,
        /// variadic tuple of the remaining elements.
        pub tail: Tail,
    }
}

/// argument type that marks the benchmark functions as plottable.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(C)]
pub struct PlotArg(pub usize);

/// benchmark configuration
pub mod config {
    use super::*;

    impl fmt::Debug for PlotArg {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.0.fmt(f)
        }
    }

    /// number of samples to use during benchmarking.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct SampleCount(pub u64);

    /// size of the plot.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct PlotSize {
        pub x: u32,
        pub y: u32,
    }

    /// kind of a plot axis.
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
    pub enum PlotAxis {
        Linear,
        SemiLogX,
        SemiLogY,
        #[default]
        LogLog,
    }

    /// metric to use for plots, default is time spent.
    pub struct PlotMetric(pub Box<dyn traits::PlotMetric>);

    /// plot output directory.
    #[derive(Clone, Debug)]
    pub struct PlotDir(pub Option<PathBuf>);

    impl Default for PlotDir {
        fn default() -> Self {
            Self(cargo_target_directory())
        }
    }

    impl Clone for PlotMetric {
        fn clone(&self) -> Self {
            Self(dyn_clone::clone_box(&*self.0))
        }
    }

    impl fmt::Debug for PlotMetric {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_tuple("PlotMetric").field(&self.0.name()).finish()
        }
    }

    impl Default for PlotMetric {
        fn default() -> Self {
            Self(Box::new(TimeMetric))
        }
    }

    impl PlotMetric {
        /// create a new metric from the given metric function.
        pub fn new(metric: impl traits::PlotMetric) -> Self {
            Self(Box::new(metric))
        }

        /// change the name of the plot metric.
        pub fn with_name(self, name: &str) -> Self {
            #[derive(Clone)]
            struct Wrap(PlotMetric, String);

            impl traits::PlotMetric for Wrap {
                fn compute(&self, arg: PlotArg, time: Picoseconds) -> f64 {
                    self.0 .0.compute(arg, time)
                }

                fn monotonicity(&self) -> Monotonicity {
                    self.0 .0.monotonicity()
                }

                fn name(&self) -> &str {
                    &self.1
                }
            }

            Self::new(Wrap(self, name.to_string()))
        }
    }

    /// number of iterations per sample.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub enum ItersPerSample {
        /// determine the number of iterations automatically.
        Auto,
        /// fixed number of iterations.
        Manual(u64),
    }

    /// minimum benchmark time.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct MinTime(pub Duration);

    /// maximum benchmark time.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct MaxTime(pub Duration);

    /// whether the benchmark results should be printed to stdout.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub enum StdoutPrint {
        /// do not print to stdout.
        Quiet,
        /// print to stdout.
        Verbose,
    }

    /// plot file name prefix.
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
            Self(Duration::from_secs(3))
        }
    }

    impl Default for StdoutPrint {
        fn default() -> Self {
            Self::Verbose
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

    /// colors to use for the generated plot
    #[derive(Debug, Clone, Default)]
    pub enum PlotColors {
        CubehelixDefault,
        Turbo,
        Spectral,
        #[default]
        Viridis,
        Magma,
        Inferno,
        Plasma,
        Cividis,
        Warm,
        Cool,
    }

    /// benchmark configuration.
    #[derive(Debug, Clone, Default)]
    pub struct BenchConfig {
        pub sample_count: SampleCount,
        pub iter_count: ItersPerSample,
        pub min_time: MinTime,
        pub max_time: MaxTime,
        pub verbose: StdoutPrint,
        pub plot_size: PlotSize,
        pub plot_axis: PlotAxis,
        pub plot_name: PlotName,
        pub plot_metric: PlotMetric,
        pub plot_dir: PlotDir,
        pub plot_colors: PlotColors,
        pub func_filter: Option<Regex>,
        pub arg_filter: Option<Regex>,
        pub output: Option<PathBuf>,
    }

    impl BenchConfig {
        /// create a default configuration
        pub fn new() -> Self {
            Default::default()
        }

        /// create a configuration from parsed program command-line arguments
        pub fn from_args() -> eyre::Result<Self> {
            let mut config = Self::default();

            #[derive(clap::ValueEnum, Debug, Clone, Serialize, Deserialize)]
            #[clap(rename_all = "kebab_case")]
            #[serde(rename_all = "kebab-case")]
            enum PlotColors {
                CubehelixDefault,
                Turbo,
                Spectral,
                Viridis,
                Magma,
                Inferno,
                Plasma,
                Cividis,
                Warm,
                Cool,
            }

            #[derive(Serialize, Deserialize)]
            struct Toml {
                sample_count: Option<u64>,
                min_time: Option<f64>,
                max_time: Option<f64>,
                quiet: Option<bool>,
                output: Option<PathBuf>,
                plot_dir: Option<PathBuf>,
                colors: Option<PlotColors>,
            }

            #[derive(Parser)]
            struct Clap {
                #[arg(long, hide(true))]
                bench: bool,

                /// config toml file.
                #[arg(long)]
                config: Option<PathBuf>,

                /// config toml file to write the config to.
                #[arg(long)]
                config_out: Option<PathBuf>,

                /// number of benchmark samples. each benchmark is run at least `sample_count *
                /// iter_count` times unless the maximum time is reached.
                #[arg(long)]
                sample_count: Option<u64>,

                /// minimum time for each benchmark.
                #[arg(long)]
                min_time: Option<f64>,

                /// maximum time for each benchmark.
                #[arg(long)]
                max_time: Option<f64>,

                /// specifies whether the output should be written to stdout.
                #[arg(long)]
                quiet: bool,

                /// regex to filter the benchmark functions.
                #[arg(long)]
                func_filter: Option<Regex>,

                /// regex to filter the benchmark arguments.
                #[arg(long)]
                arg_filter: Option<Regex>,

                /// output file.
                #[arg(long)]
                output: Option<PathBuf>,

                /// plot directory.
                #[arg(long)]
                plot_dir: Option<PathBuf>,

                #[arg(long)]
                colors: Option<PlotColors>,
            }

            let clap = Clap::parse();

            let toml: Option<eyre::Result<Toml>> = clap.config.map(|toml| -> eyre::Result<Toml> {
                Ok(toml::de::from_str(&std::fs::read_to_string(toml)?)?)
            });
            let toml = match toml {
                Some(Ok(toml)) => Some(toml),
                Some(Err(e)) => return Err(e),
                None => None,
            }
            .unwrap_or(Toml {
                sample_count: None,
                min_time: None,
                max_time: None,
                quiet: None,
                output: None,
                plot_dir: None,
                colors: None,
            });

            if let (Some(sample_count), _) | (None, Some(sample_count)) =
                (clap.sample_count, toml.sample_count)
            {
                config.sample_count = SampleCount(sample_count)
            }

            if let (Some(min_time), _) | (None, Some(min_time)) = (clap.min_time, toml.min_time) {
                config.min_time = MinTime(Duration::from_secs_f64(min_time))
            }
            if let (Some(max_time), _) | (None, Some(max_time)) = (clap.max_time, toml.max_time) {
                config.max_time = MaxTime(Duration::from_secs_f64(max_time))
            }
            if let (Some(plot_dir), _) | (None, Some(plot_dir)) = (clap.plot_dir, toml.plot_dir) {
                config.plot_dir = PlotDir(Some(plot_dir));
            }
            if clap.quiet || toml.quiet == Some(true) {
                config.verbose = StdoutPrint::Quiet;
            }
            if let (Some(output), _) | (None, Some(output)) = (clap.output, toml.output) {
                config.output = Some(output);
            };
            if let (Some(colors), _) | (None, Some(colors)) = (clap.colors, toml.colors) {
                config.plot_colors = match colors {
                    PlotColors::CubehelixDefault => crate::PlotColors::CubehelixDefault,
                    PlotColors::Turbo => crate::PlotColors::Turbo,
                    PlotColors::Spectral => crate::PlotColors::Spectral,
                    PlotColors::Viridis => crate::PlotColors::Viridis,
                    PlotColors::Magma => crate::PlotColors::Magma,
                    PlotColors::Inferno => crate::PlotColors::Inferno,
                    PlotColors::Plasma => crate::PlotColors::Plasma,
                    PlotColors::Cividis => crate::PlotColors::Cividis,
                    PlotColors::Warm => crate::PlotColors::Warm,
                    PlotColors::Cool => crate::PlotColors::Cool,
                };
            }

            if let Some(func_filter) = clap.func_filter {
                config.func_filter = Some(func_filter);
            }
            if let Some(arg_filter) = clap.arg_filter {
                config.arg_filter = Some(arg_filter);
            }

            if let Some(config_out) = clap.config_out {
                let toml = Toml {
                    sample_count: Some(config.sample_count.0),
                    min_time: Some(config.min_time.0.as_secs_f64()),
                    max_time: Some(config.max_time.0.as_secs_f64()),
                    quiet: Some(config.verbose == StdoutPrint::Quiet),
                    output: config.output.clone(),
                    plot_dir: config.plot_dir.0.clone(),
                    colors: Some(match config.plot_colors {
                        crate::PlotColors::CubehelixDefault => PlotColors::CubehelixDefault,
                        crate::PlotColors::Turbo => PlotColors::Turbo,
                        crate::PlotColors::Spectral => PlotColors::Spectral,
                        crate::PlotColors::Viridis => PlotColors::Viridis,
                        crate::PlotColors::Magma => PlotColors::Magma,
                        crate::PlotColors::Inferno => PlotColors::Inferno,
                        crate::PlotColors::Plasma => PlotColors::Plasma,
                        crate::PlotColors::Cividis => PlotColors::Cividis,
                        crate::PlotColors::Warm => PlotColors::Warm,
                        crate::PlotColors::Cool => PlotColors::Cool,
                    }),
                };
                let toml = toml::ser::to_string_pretty(&toml)?;
                std::fs::write(config_out, toml)?;
            }

            Ok(config)
        }
    }
}

/// benchmark result output.
pub mod result {
    pub use self::traits::Monotonicity;

    use super::*;

    #[derive(Debug)]
    #[repr(transparent)]
    pub struct Stats<T>(pub [T]);

    impl<T> std::ops::Deref for Stats<T> {
        type Target = [T];

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<T> std::ops::DerefMut for Stats<T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    impl<T> Stats<T> {
        pub fn from_slice(slice: &[T]) -> &Self {
            unsafe { &*(slice as *const [T] as *const Self) }
        }

        pub fn from_slice_mut(slice: &mut [T]) -> &mut Self {
            unsafe { &mut *(slice as *mut [T] as *mut Self) }
        }
    }

    impl Stats<Picoseconds> {
        pub fn mean_stddev(&self) -> (Picoseconds, Picoseconds) {
            if self.len() == 0 {
                return (Picoseconds(0), Picoseconds(0));
            }

            let count = self.len();
            let sum = self.0.iter().copied().sum::<Picoseconds>();
            let mean = sum / count as i128;

            let variance = if count <= 1 {
                0
            } else {
                self.0
                    .iter()
                    .map(|x| {
                        let diff = x.0 - mean.0;
                        diff * diff
                    })
                    .sum::<i128>()
                    / (count as i128 - 1)
            };
            let stddev = Picoseconds(isqrt(variance as u128) as i128);

            (mean, stddev)
        }
    }

    impl Stats<f64> {
        pub fn mean_stddev(&self) -> (f64, f64) {
            if self.len() == 0 {
                return (f64::NAN, f64::NAN);
            }

            let count = self.len();
            let sum = self.0.iter().copied().sum::<f64>();
            let mean = sum / count as f64;

            let variance = if count <= 1 {
                0.0
            } else {
                self.0
                    .iter()
                    .map(|x| {
                        let diff = x - mean;
                        diff * diff
                    })
                    .sum::<f64>()
                    / ((count - 1) as f64)
            };
            let stddev = variance.sqrt();

            (mean, stddev)
        }
    }

    /// single benchmark argument.
    #[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub enum BenchArg<'a> {
        /// generic argument
        Named(&'a str),
        /// plot argument
        Plot(PlotArg),
    }

    /// benchmark arguments.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
    pub enum BenchArgs {
        /// generic argument
        Named(Vec<String>),
        /// plot argument
        Plot(Vec<PlotArg>),
    }

    impl BenchArgs {
        /// number of arguments
        pub fn len(&self) -> usize {
            match self {
                BenchArgs::Named(a) => a.len(),
                BenchArgs::Plot(a) => a.len(),
            }
        }

        #[must_use]
        pub fn is_empty(&self) -> bool {
            self.len() == 0
        }

        #[track_caller]
        pub fn unwrap_as_named(&self) -> &[String] {
            match self {
                BenchArgs::Named(this) => this,
                BenchArgs::Plot(_) => panic!(),
            }
        }

        #[track_caller]
        pub fn unwrap_as_plot_arg(&self) -> &[PlotArg] {
            match self {
                BenchArgs::Named(_) => panic!(),
                BenchArgs::Plot(this) => this,
            }
        }
    }

    /// benchmark function timing results.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BenchFunctionResult {
        /// function name.
        pub name: String,
        /// timings for each argument, given in the same order as the vector in [`BenchArgs`].
        pub timings: Vec<Vec<Picoseconds>>,
        /// computed plot metric, if the argument type is [`PlotArg`], otherwise `None`.
        pub metric: Option<Vec<Vec<f64>>>,
    }

    /// benchmark result of a single group.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BenchGroupResult {
        /// results of benchmark function timings.
        pub function: Vec<BenchFunctionResult>,
        /// benchmark arguments.
        pub args: BenchArgs,
        /// metric name for plotting.
        pub metric_name: String,
        pub metric_mono: Monotonicity,
    }

    /// benchmark result of all groups.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BenchResult {
        pub groups: Vec<BenchGroupResult>,
    }

    #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Group(pub usize);
    #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Func(pub usize);
    #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Arg(pub usize);

    impl BenchFunctionResult {
        /// get the timings and metrics for the given argument index
        #[track_caller]
        pub fn at(&self, Arg(arg_idx): Arg) -> (&Stats<Picoseconds>, Option<&Stats<f64>>) {
            equator::assert!(arg_idx < self.timings.len());

            (
                Stats::from_slice(&self.timings[arg_idx]),
                self.metric
                    .as_ref()
                    .map(|metric| Stats::from_slice(&metric[arg_idx])),
            )
        }
    }

    impl BenchGroupResult {
        /// get the timings and metrics for the given function and argument index
        #[track_caller]
        pub fn at(
            &self,
            Func(func_idx): Func,
            Arg(arg_idx): Arg,
        ) -> (&Stats<Picoseconds>, Option<&Stats<f64>>) {
            equator::assert!(all(
                func_idx < self.function.len(),
                arg_idx < self.args.len(),
            ));
            self.function[func_idx].at(Arg(arg_idx))
        }

        #[track_caller]
        pub fn arg(&self, i: usize) -> BenchArg<'_> {
            match &self.args {
                BenchArgs::Named(name) => BenchArg::Named(&name[i]),
                BenchArgs::Plot(arg) => BenchArg::Plot(arg[i]),
            }
        }

        #[cfg(feature = "plot")]
        pub fn plot(
            &self,
            plot_name: &str,
            axis: PlotAxis,
            dir: &std::path::Path,
        ) -> eyre::Result<()> {
            let group = self;
            let mut code = String::new();
            let plot_svg = dir.join(format!("{plot_name}.svg"));
            let plot_pdf = dir.join(format!("{plot_name}.pdf"));

            let mut max_y = f64::NEG_INFINITY;
            let mut min_y = f64::INFINITY;
            let metric_name = &group.metric_name;

            match &group.args {
                BenchArgs::Plot(args) => {
                    let mut args = args
                        .iter()
                        .map(|arg| (false, arg.0 as f64))
                        .collect::<Vec<_>>();

                    let nfuncs = group.function.len();

                    for (idx, f) in group.function.iter().enumerate() {
                        let name = &f.name;
                        let mut line = String::new();

                        for (arg_idx, (keep, arg)) in args.iter_mut().enumerate() {
                            let metric = &*f.metric.as_ref().unwrap()[arg_idx];
                            let metric_mean = result::Stats::from_slice(&metric).mean_stddev().0;
                            if metric_mean.is_finite() {
                                max_y = f64::max(max_y, metric_mean);
                                min_y = f64::min(min_y, metric_mean);

                                *keep = true;
                                let arg = if matches!(axis, PlotAxis::LogLog | PlotAxis::SemiLogX) {
                                    (*arg).log2()
                                } else {
                                    *arg
                                };
                                line += &format!("({arg}, {metric_mean}),");
                            }
                        }
                        if !line.is_empty() {
                            let gradient = colorgrad::spectral();
                            let color = gradient.at(if nfuncs == 0 {
                                0.5
                            } else {
                                idx as f64 / (nfuncs - 1) as f64
                            });
                            let r = (color.r * 255.0) as u8;
                            let g = (color.g * 255.0) as u8;
                            let b = (color.b * 255.0) as u8;

                            code += &format!(
                                "
plot.add(
    ({line}),
    line: \"linear\",
    label: \"{name}\",
    mark: \"o\",
    style: (
        stroke: (
            thickness: 2pt,
            dash: \"solid\",
            paint: rgb(\"#{r:02x}{g:02x}{b:02x}\"),
        ),
    ),
    mark-style: (
        stroke: rgb(\"#{r:02x}{g:02x}{b:02x}\"),
        fill: rgb(\"#{r:02x}{g:02x}{b:02x}\"),
    ),
)
"
                            );
                        }
                    }

                    let args = args
                        .iter()
                        .filter(|(keep, _)| *keep)
                        .map(|(_, arg)| *arg)
                        .collect::<Vec<_>>();

                    let ticks = args
                        .iter()
                        .map(|arg| format!("{arg}"))
                        .collect::<Vec<_>>()
                        .join(",");

                    let xmin = args[0];
                    let xmax = *args.last().unwrap();

                    let (xmin, xmax, ticks) = if matches!(
                        axis,
                        PlotAxis::LogLog | PlotAxis::SemiLogX
                    ) {
                        (
                            xmin.log2(),
                            xmax.log2(),
                            format!("#let ticks = ({ticks},).map((i) => (calc.log(i, base: 2), rotate(-45deg, reflow: true)[#i]));"),
                        )
                    } else {
                        (
                            xmin,
                            xmax,
                            format!("#let ticks = ({ticks},).map((i) => (i, rotate(-45deg, reflow: true)[#i]));"),
                        )
                    };

                    let log = if matches!(axis, PlotAxis::LogLog | PlotAxis::SemiLogY) {
                        "log"
                    } else {
                        "lin"
                    };

                    let source = format!(
                        r###"
#import "@preview/cetz:0.3.4"
#import "@preview/cetz-plot:0.1.1"

#set text(14pt)
#align(center + horizon)[

{ticks}

#cetz.canvas({{
import cetz.draw: *
import cetz-plot: *

plot.plot(size: (16,8),
    x-format: plot.formats.sci,
    y-format: plot.formats.sci,
    legend: (0.5, 7.5),
    legend-anchor: "north-west",
    x-mode: "lin", y-mode: "{log}", y-base: 2,
    y-grid: true,
    x-min: {xmin}, x-max: {xmax},
    y-min: {min_y} / 1.125, y-max: {max_y} * 1.125,
    x-ticks: ticks,
    x-tick-step: none, y-tick-step: 1.0, y-minor-tick-step: 0.2, 
    x-label: "input", y-label: "{metric_name}",
{{
    {code}
}})
}})
]
"###
                    );
                    use typst_imp::*;
                    let (svg, pdf) = TypstCompiler::new().compile(&source).unwrap();
                    std::fs::write(plot_svg, svg)?;
                    std::fs::write(plot_pdf, pdf)?;
                }
                _ => {}
            }
            Ok(())
        }
    }

    impl BenchResult {
        /// get the timings and metrics for the given group, function and argument index
        #[track_caller]
        pub fn at(
            &self,
            Group(group_idx): Group,
            Func(func_idx): Func,
            Arg(arg_idx): Arg,
        ) -> (&Stats<Picoseconds>, Option<&Stats<f64>>) {
            equator::assert!(group_idx < self.groups.len());
            self.groups[group_idx].at(Func(func_idx), Arg(arg_idx))
        }

        pub fn combine(&self, other: &Self) -> Self {
            BenchResult {
                groups: core::iter::zip(&self.groups, &other.groups)
                    .map(|(left, right)| {
                        assert_eq!(left.args, right.args);
                        assert_eq!(left.metric_name, right.metric_name);
                        assert_eq!(left.metric_mono, right.metric_mono);

                        let mut set = std::collections::HashSet::new();

                        for f in &left.function {
                            set.insert(&*f.name);
                        }
                        let mut function = left.function.clone();
                        for f in &right.function {
                            if !set.contains(&*f.name) {
                                function.push(f.clone());
                            }
                        }

                        BenchGroupResult {
                            function,
                            args: left.args.clone(),
                            metric_name: left.metric_name.clone(),
                            metric_mono: left.metric_mono.clone(),
                        }
                    })
                    .collect(),
            }
        }

        #[cfg(feature = "plot")]
        pub fn plot(
            &self,
            plot_name: &str,
            axis: PlotAxis,
            dir: &std::path::Path,
        ) -> eyre::Result<()> {
            for (plot_id, group) in self.groups.iter().enumerate() {
                group.plot(&format!("{plot_name}_{plot_id}.svg"), axis, dir)?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[derive(Clone)]
    pub struct BytesPerSecMetric;
    impl traits::PlotMetric for BytesPerSecMetric {
        fn name(&self) -> &'static str {
            "bytes/s"
        }
        fn compute(&self, arg: PlotArg, time: Picoseconds) -> f64 {
            arg.0 as f64 / time.to_secs()
        }
        fn monotonicity(&self) -> traits::Monotonicity {
            traits::Monotonicity::HigherIsBetter
        }
    }

    #[test]
    fn test_list() {
        let unlist![_, _, _,]: List![i32, u32, usize] = list![1, 2, 3,];
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
        let bench = Bench::new(BenchConfig {
            plot_axis: PlotAxis::LogLog,
            min_time: MinTime(Duration::from_millis(100)),
            max_time: MaxTime(Duration::from_millis(100)),
            arg_filter: Some(Regex::new("1").unwrap()),
            output: cargo_target_directory().map(|x| x.join("diol.json")),
            ..Default::default()
        });
        println!();
        bench.register_many(
            list![naive.with_name("naive loop"), popcnt],
            (0..20).map(|i| 1 << i).map(PlotArg),
        );
        bench.run().unwrap();
    }
}
