// code from https://github.com/cestef/zola

use eyre::Result;
use std::{collections::HashMap, io::Write, path::PathBuf, sync::Mutex};
use typst::layout::PagedDocument;

use typst::{
    diag::{eco_format, FileError, FileResult, PackageError, PackageResult},
    foundations::{Bytes, Datetime},
    syntax::{package::PackageSpec, FileId, Source},
    text::{Font, FontBook},
    utils::LazyHash,
    Library, World,
};

mod templates;

pub trait MathCompiler {
    fn compile(&self, input: &str) -> Result<(String, Vec<u8>)>;
}

use crate::cargo_target_directory;

fn fonts() -> Vec<Font> {
    typst_assets::fonts()
        .flat_map(|bytes| {
            let buffer = Bytes::new(bytes);
            let face_count = ttf_parser::fonts_in_collection(&buffer).unwrap_or(1);
            (0..face_count).map(move |face| {
                Font::new(buffer.clone(), face).expect("failed to load font from typst-assets")
            })
        })
        .collect()
}

/// Fake file
///
/// This is a fake file which wrap the real content takes from the md math block
pub struct TypstFile {
    bytes: Bytes,

    source: Option<Source>,
}

impl TypstFile {
    fn source(&mut self, id: FileId) -> FileResult<Source> {
        let source = match &self.source {
            Some(source) => source,
            None => {
                let contents =
                    std::str::from_utf8(&self.bytes).map_err(|_| FileError::InvalidUtf8)?;
                let source = Source::new(id, contents.into());
                self.source.insert(source)
            }
        };
        Ok(source.clone())
    }
}

/// Compiler
///
/// This is the compiler which has all the necessary fields except the source
pub struct TypstCompiler {
    library: LazyHash<Library>,
    book: LazyHash<FontBook>,
    fonts: Vec<Font>,
    packages_cache_path: PathBuf,
    files: Mutex<HashMap<FileId, TypstFile>>,
}

impl TypstCompiler {
    pub fn new() -> Self {
        let fonts = fonts();

        Self {
            library: LazyHash::new(Library::default()),
            book: LazyHash::new(FontBook::from_fonts(&fonts)),
            fonts,
            packages_cache_path: cargo_target_directory()
                .unwrap()
                .to_path_buf()
                .join("typst-packages"),
            files: Mutex::new(HashMap::new()),
        }
    }

    pub fn wrap_source(&self, source: impl Into<String>) -> WrapSource<'_> {
        WrapSource {
            compiler: self,
            source: Source::detached(source),
            time: time::OffsetDateTime::now_local().unwrap_or(time::OffsetDateTime::now_utc()),
        }
    }

    /// Get the package directory or download if not exists
    fn package(&self, package: &PackageSpec) -> PackageResult<PathBuf> {
        let package_subdir = format!("{}/{}/{}", package.namespace, package.name, package.version);
        let path = self.packages_cache_path.join(package_subdir);

        if path.exists() {
            return Ok(path);
        }

        // Download the package
        let package_url = format!(
            "https://packages.typst.org/{}/{}-{}.tar.gz",
            package.namespace, package.name, package.version
        );

        let mut response = reqwest::blocking::get(package_url).map_err(|e| {
            PackageError::NetworkFailed(Some(eco_format!(
                "Failed to download package {}: {}",
                package.name,
                e
            )))
        })?;

        let mut compressed = Vec::new();
        response.copy_to(&mut compressed).map_err(|e| {
            PackageError::NetworkFailed(Some(eco_format!(
                "Failed to save package {}: {}",
                package.name,
                e
            )))
        })?;

        let mut decompressed = Vec::new();
        let mut decoder = flate2::write::GzDecoder::new(decompressed);
        decoder.write_all(&compressed).map_err(|e| {
            PackageError::MalformedArchive(Some(eco_format!(
                "Failed to decompress package {}: {}",
                package.name,
                e
            )))
        })?;
        decoder.try_finish().map_err(|e| {
            PackageError::MalformedArchive(Some(eco_format!(
                "Failed to decompress package {}: {}",
                package.name,
                e
            )))
        })?;
        decompressed = decoder.finish().map_err(|e| {
            PackageError::MalformedArchive(Some(eco_format!(
                "Failed to decompress package {}: {}",
                package.name,
                e
            )))
        })?;

        let mut archive = tar::Archive::new(decompressed.as_slice());
        archive.unpack(&path).map_err(|e| {
            std::fs::remove_dir_all(&path).ok();
            PackageError::MalformedArchive(Some(eco_format!(
                "Failed to unpack package {}: {}",
                package.name,
                e
            )))
        })?;

        Ok(path)
    }

    // Weird pattern because mapping a MutexGuard is not stable yet.
    fn file<T>(&self, id: FileId, map: impl FnOnce(&mut TypstFile) -> T) -> FileResult<T> {
        let mut files = self.files.lock().unwrap();
        if let Some(entry) = files.get_mut(&id) {
            return Ok(map(entry));
        }
        // `files` must stay locked here so we don't download the same package multiple times.
        // TODO proper multithreading, maybe with typst-kit.

        'x: {
            if let Some(package) = id.package() {
                let package_dir = self.package(package)?;
                let Some(path) = id.vpath().resolve(&package_dir) else {
                    break 'x;
                };
                let contents =
                    std::fs::read(&path).map_err(|error| FileError::from_io(error, &path))?;
                let entry = files.entry(id).or_insert(TypstFile {
                    bytes: Bytes::new(contents),
                    source: None,
                });
                return Ok(map(entry));
            }
        }

        Err(FileError::NotFound(id.vpath().as_rootless_path().into()))
    }
}

impl MathCompiler for TypstCompiler {
    fn compile(&self, source: &str) -> Result<(String, Vec<u8>)> {
        // Prepare source based on mode
        let source = templates::raw(&source);

        // Compile the source
        let world = self.wrap_source(source);
        let document = typst::compile(&world);
        let warnings = document.warnings;
        let has_error = warnings
            .iter()
            .any(|w| w.severity == typst::diag::Severity::Error);
        if has_error {
            return Err(eyre::Report::msg(format!("{:?}", warnings)));
        }

        let document: PagedDocument = document
            .output
            .map_err(|diags| eyre::Report::msg(format!("{:?}", diags)))?;
        let page = document
            .pages
            .first()
            .ok_or(eyre::Report::msg("No pages found"))?;

        Ok((
            typst_svg::svg(page),
            typst_pdf::pdf(
                &PagedDocument {
                    pages: vec![page.clone()],
                    info: Default::default(),
                    introspector: Default::default(),
                },
                &Default::default(),
            )
            .unwrap(),
        ))
    }
}

/// Wrap source
///
/// This is a wrapper for the source which provides ref to the compiler
pub struct WrapSource<'a> {
    compiler: &'a TypstCompiler,
    source: Source,
    time: time::OffsetDateTime,
}

impl World for WrapSource<'_> {
    fn library(&self) -> &LazyHash<Library> {
        &self.compiler.library
    }

    fn book(&self) -> &LazyHash<FontBook> {
        &self.compiler.book
    }

    fn main(&self) -> FileId {
        self.source.id()
    }

    fn source(&self, id: FileId) -> FileResult<Source> {
        if id == self.source.id() {
            Ok(self.source.clone())
        } else {
            self.compiler.file(id, |file| file.source(id))?
        }
    }

    fn file(&self, id: FileId) -> FileResult<Bytes> {
        self.compiler.file(id, |file| file.bytes.clone())
    }

    fn font(&self, index: usize) -> Option<Font> {
        self.compiler.fonts.get(index).cloned()
    }

    fn today(&self, _offset: Option<i64>) -> Option<Datetime> {
        Some(Datetime::Date(self.time.date()))
    }
}
