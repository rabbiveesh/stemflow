//! Core ML models run directly rather than through ONNX Runtime, for graphs ORT's
//! Core ML EP handles badly (BS-RoFormer: tools/export_bsrofo_coreml.py). One f32
//! input, one f32 output, fixed shapes.

use anyhow::{anyhow, bail, Context, Result};
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::{AnyObject, ProtocolObject};
use objc2::AnyThread;
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{NSArray, NSDictionary, NSError, NSNumber, NSString, NSURL};
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

pub struct Model {
    model: Retained<MLModel>,
    input: Retained<NSString>,
    output: Retained<NSString>,
}

// Core ML models may be used from any thread (one prediction at a time here).
unsafe impl Send for Model {}

fn ns_err(e: Retained<NSError>) -> anyhow::Error {
    anyhow!("{}", e.localizedDescription())
}

fn url(p: &Path) -> Retained<NSURL> {
    NSURL::fileURLWithPath(&NSString::from_str(&p.to_string_lossy()))
}

/// The compiled `.mlmodelc` for a `.mlpackage`, compiled on first use (and again when
/// the package is newer) and kept next to it.
fn compiled(pkg: &Path) -> Result<PathBuf> {
    let out = pkg.with_extension("mlmodelc");
    let mtime = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    if out.is_dir() && mtime(&out) >= mtime(&pkg.join("Manifest.json")) {
        return Ok(out);
    }
    eprintln!("stemflow: compiling {} for Core ML", pkg.display());
    #[allow(deprecated)] // the async variant needs a run loop; this runs once
    let tmp = unsafe { MLModel::compileModelAtURL_error(&url(pkg)) }.map_err(ns_err)?;
    let tmp = PathBuf::from(
        tmp.path()
            .context("compiled model has no path")?
            .to_string(),
    );
    let _ = std::fs::remove_dir_all(&out);
    Ok(match std::fs::rename(&tmp, &out) {
        Ok(()) => out,
        Err(_) => tmp, // another volume: use it where it is, compile again next time
    })
}

impl Model {
    /// Load on the CPU and GPU: the Neural Engine is much slower for transformers
    /// like BS-RoFormer's (2.6 s vs 0.31 s per 3 s window on an M5).
    pub fn load(pkg: &Path, input: &str, output: &str) -> Result<Model> {
        if !pkg.is_dir() {
            bail!("no Core ML package at {}", pkg.display());
        }
        let c = compiled(pkg)?;
        let model = unsafe {
            let config = MLModelConfiguration::new();
            config.setComputeUnits(MLComputeUnits::CPUAndGPU);
            MLModel::modelWithContentsOfURL_configuration_error(&url(&c), &config)
        }
        .map_err(ns_err)
        .with_context(|| format!("loading {}", c.display()))?;
        Ok(Model {
            model,
            input: NSString::from_str(input),
            output: NSString::from_str(output),
        })
    }

    /// Run on a row-major `shape` input; returns the output, row-major.
    pub fn run(&self, shape: &[usize], data: &mut [f32]) -> Result<Vec<f32>> {
        assert_eq!(shape.iter().product::<usize>(), data.len());
        autoreleasepool(|_| unsafe {
            let mut strides = vec![1usize; shape.len()];
            for i in (0..shape.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            let nums = |v: &[usize]| {
                NSArray::from_retained_slice(
                    &v.iter()
                        .map(|&n| NSNumber::new_isize(n as isize))
                        .collect::<Vec<_>>(),
                )
            };
            // Wraps `data` without copying; it outlives the prediction.
            let arr = MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
                MLMultiArray::alloc(),
                NonNull::new(data.as_mut_ptr() as *mut c_void).unwrap(),
                &nums(shape),
                MLMultiArrayDataType::Float32,
                &nums(&strides),
                None,
            )
            .map_err(ns_err)?;
            let value = MLFeatureValue::featureValueWithMultiArray(&arr);
            let value: &AnyObject = &value;
            let dict = NSDictionary::from_slices(&[&*self.input], &[value]);
            let features = MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                &dict,
            )
            .map_err(ns_err)?;
            let out = self
                .model
                .predictionFromFeatures_error(ProtocolObject::from_ref(&*features))
                .map_err(ns_err)?;
            let out = out
                .featureValueForName(&self.output)
                .and_then(|v| v.multiArrayValue())
                .with_context(|| format!("no output `{}`", self.output))?;
            read(&out)
        })
    }
}

/// A multi-array's values in row-major order, whatever its strides (Core ML pads
/// some outputs) and element type.
unsafe fn read(a: &MLMultiArray) -> Result<Vec<f32>> {
    let shape: Vec<usize> = a.shape().iter().map(|n| n.as_isize() as usize).collect();
    let strides: Vec<usize> = a.strides().iter().map(|n| n.as_isize() as usize).collect();
    let n: usize = shape.iter().product();
    let dt = a.dataType();
    #[allow(deprecated)] // getBytesWithHandler wants a block for the same pointer
    let p = a.dataPointer().as_ptr();
    let get: Box<dyn Fn(usize) -> f32> = if dt == MLMultiArrayDataType::Float32 {
        let p = p as *const f32;
        Box::new(move |o| *p.add(o))
    } else if dt == MLMultiArrayDataType::Float16 {
        let p = p as *const u16;
        Box::new(move |o| f16(*p.add(o)))
    } else {
        bail!("Core ML output type {:#x} isn't f32 or f16", dt.0);
    };
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; shape.len()];
    let mut off = 0usize;
    for _ in 0..n {
        out.push(get(off));
        // Odometer increment over the index, keeping the strided offset in step.
        for d in (0..shape.len()).rev() {
            idx[d] += 1;
            off += strides[d];
            if idx[d] < shape[d] {
                break;
            }
            off -= strides[d] * shape[d];
            idx[d] = 0;
        }
    }
    Ok(out)
}

fn f16(h: u16) -> f32 {
    let (s, e, m) = (
        (h >> 15) as u32,
        ((h >> 10) & 0x1f) as u32,
        (h & 0x3ff) as u32,
    );
    let bits = match e {
        0 if m == 0 => s << 31,
        0 => {
            // subnormal: normalise
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (s << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => (s << 31) | (0xff << 23) | (m << 13),
        _ => (s << 31) | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    #[test]
    fn f16() {
        for (h, f) in [
            (0x3c00, 1.0),
            (0xc000, -2.0),
            (0x0001, 5.960_464_5e-8),
            (0x7bff, 65504.0),
        ] {
            assert_eq!(super::f16(h), f);
        }
    }
}
