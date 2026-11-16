//! The models stemflow can switch between live, found by scanning the models folder.
//!
//! A model is an exported `.onnx` file, either at the top of the folder or one level
//! down (`models/<experiment>/<file>.onnx`, e.g. a symlink to a fine-tune's export
//! folder). The file name says what it is: the prefix picks the architecture
//! (`bsrofo_`, `htdemucs`, `scnet_`) and the suffix names the stem it isolates
//! (`_vocals`, `_other`, ...; anything else is a bass model). The id is the path relative to the folder without `.onnx`, e.g.
//! `bsrofo_sw_bass` or `ft2/htdemucs_ft_bass`. Dropping a new file in shows up in the
//! menu the next time it opens; nothing needs restarting.

use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    Bsrofo,
    Demucs,
    Scnet,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub id: String,
    pub label: String,
    pub arch: Arch,
    /// The stem it isolates: one of `STEMS`.
    pub target: &'static str,
    pub path: PathBuf,
}

/// The stems a model can target, in menu order. "other" is what's left once the
/// rest are taken out: in a jazz combo, the horns.
pub const STEMS: [&str; 6] = ["bass", "vocals", "other", "piano", "guitar", "drums"];

fn classify(stem: &str) -> Option<(Arch, &'static str, &'static str)> {
    let (arch, name) = if stem.starts_with("bsrofo_sw") {
        (Arch::Bsrofo, "BS-RoFormer SW")
    } else if stem.starts_with("bsrofo") {
        (Arch::Bsrofo, "BS-RoFormer")
    } else if stem.starts_with("htdemucs") {
        (Arch::Demucs, "Demucs ft")
    } else if stem.starts_with("scnet") {
        (Arch::Scnet, "SCNet XL")
    } else {
        return None;
    };
    let target = STEMS
        .into_iter()
        .find(|t| stem.strip_suffix(t).is_some_and(|s| s.ends_with('_')))
        .unwrap_or("bass");
    Some((arch, target, name))
}

fn model(dir: &Path, rel: &Path) -> Option<Model> {
    if rel.extension()? != "onnx" {
        return None;
    }
    let stem = rel.file_stem()?.to_str()?;
    let (arch, target, name) = classify(stem)?;
    let id = rel.with_extension("").to_str()?.to_string();
    let label = match rel
        .parent()
        .and_then(|p| p.to_str())
        .filter(|p| !p.is_empty())
    {
        Some(exp) => format!("{exp}: {name}"),
        None => name.to_string(),
    };
    let label = if target == "bass" {
        label
    } else {
        format!("{label} ({target})")
    };
    Some(Model {
        id,
        label,
        arch,
        target,
        path: dir.join(rel),
    })
}

/// Every model in `dir`, by stem (bass models first), then built-ins before experiments.
pub fn scan(dir: &Path) -> Vec<Model> {
    let mut out = Vec::new();
    let entries = |p: &Path| -> Vec<std::fs::DirEntry> {
        std::fs::read_dir(p)
            .map(|r| r.flatten().collect())
            .unwrap_or_default()
    };
    for e in entries(dir) {
        let name = PathBuf::from(e.file_name());
        let path = e.path();
        if path.is_dir() {
            let sub = name.to_string_lossy();
            // reference vectors and third-party checkouts, not experiments
            if sub.starts_with("ref") || sub == "ext" {
                continue;
            }
            for f in entries(&path) {
                out.extend(model(dir, &name.join(f.file_name())));
            }
        } else {
            out.extend(model(dir, &name));
        }
    }
    let rank = |t: &str| STEMS.iter().position(|s| *s == t);
    out.sort_by_key(|m| (rank(m.target), m.id.contains('/'), m.id.clone()));
    out
}

pub fn find(dir: &Path, id: &str) -> Option<Model> {
    scan(dir).into_iter().find(|m| m.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        let d = Path::new("/m");
        let m = model(d, Path::new("ft2/htdemucs_ft_bass.onnx")).unwrap();
        assert_eq!(
            (m.id.as_str(), m.label.as_str(), m.arch, m.target),
            (
                "ft2/htdemucs_ft_bass",
                "ft2: Demucs ft",
                Arch::Demucs,
                "bass"
            )
        );
        let m = model(d, Path::new("bsrofo_sw_vocals.onnx")).unwrap();
        assert_eq!(
            (m.label.as_str(), m.target),
            ("BS-RoFormer SW (vocals)", "vocals")
        );
        let m = model(d, Path::new("bsrofo_sw_other.onnx")).unwrap();
        assert_eq!(
            (m.label.as_str(), m.target),
            ("BS-RoFormer SW (other)", "other")
        );
        // not a stem suffix: "another" ends in "other" but not "_other"
        assert_eq!(model(d, Path::new("scnet_another.onnx")).unwrap().target, "bass");
        assert!(model(d, Path::new("scnet_xl_ihf.ckpt")).is_none());
        assert!(model(d, Path::new("other.onnx")).is_none());
    }
}
