use crate::Result;
use cuda_async::device_operation::DeviceOp;
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::Tensor;
use firepit_kernels::weights::DenseMatrix;
use half::bf16;
use safetensors::Dtype;
use safetensors::tensor::{Metadata, TensorInfo};
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

pub type Stream = Arc<cuda_core::Stream>;

pub struct HostTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct HostI32Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<i32>,
}

pub struct HostFp8Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f8e4m3fn>,
}

pub struct HostNvfp4Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f4e2m1fnx2>,
}

#[derive(Clone)]
pub struct HostBf16Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<bf16>,
}

impl HostBf16Tensor {
    pub fn upload_matrix(self, stream: &Stream) -> Result<DenseMatrix<bf16>> {
        let (&cols, leading) = self.shape.split_last().ok_or("matrix has no dimensions")?;
        if leading.is_empty() {
            return Err("matrix must have at least two dimensions".into());
        }
        let rows = leading.iter().product();
        Ok(DenseMatrix::upload(self.data, rows, cols, stream)?)
    }
}

pub struct SafetensorsFile {
    file: File,
    data_start: u64,
    header: Metadata,
}

impl SafetensorsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let mut len = [0u8; 8];
        file.read_exact_at(&mut len, 0)?;
        let len = u64::from_le_bytes(len);
        let mut json = vec![0u8; len as usize];
        file.read_exact_at(&mut json, 8)?;
        Ok(Self {
            file,
            data_start: 8 + len,
            header: serde_json::from_slice(&json)?,
        })
    }

    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.header
            .metadata()
            .as_ref()?
            .get(key)
            .map(String::as_str)
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.header.info(name).is_some()
    }

    pub fn tensor(&self, name: &str) -> Result<HostTensor> {
        let info = self.info(name)?;
        let len = info.shape.iter().product();
        Ok(HostTensor {
            shape: info.shape.clone(),
            data: self.read(info, 0, len)?,
        })
    }

    pub fn tensor_i32(&self, name: &str) -> Result<HostI32Tensor> {
        let (shape, bytes) = self.bytes(name, Dtype::I32)?;
        Ok(HostI32Tensor {
            shape,
            data: bytes
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        })
    }

    pub fn tensor_bf16(&self, name: &str) -> Result<HostBf16Tensor> {
        let (shape, bytes) = self.bytes(name, Dtype::BF16)?;
        Ok(HostBf16Tensor {
            shape,
            data: bytes
                .chunks_exact(2)
                .map(|b| bf16::from_le_bytes([b[0], b[1]]))
                .collect(),
        })
    }

    pub fn tensor_fp8(&self, name: &str) -> Result<HostFp8Tensor> {
        let (shape, bytes) = self.bytes(name, Dtype::F8_E4M3)?;
        Ok(HostFp8Tensor {
            shape,
            data: bytes.into_iter().map(f8e4m3fn).collect(),
        })
    }

    pub fn tensor_nvfp4(&self, name: &str) -> Result<HostNvfp4Tensor> {
        let (shape, bytes) = self.bytes(name, Dtype::U8)?;
        Ok(HostNvfp4Tensor {
            shape,
            data: bytes.into_iter().map(f4e2m1fnx2::from_bits).collect(),
        })
    }

    pub fn scalar_f32(&self, name: &str) -> Result<f32> {
        let (shape, bytes) = self.bytes(name, Dtype::F32)?;
        match (shape.as_slice(), bytes.as_slice()) {
            ([], &[a, b, c, d]) => Ok(f32::from_le_bytes([a, b, c, d])),
            _ => Err(format!("{name}: expected an F32 scalar, got shape {shape:?}").into()),
        }
    }

    fn bytes(&self, name: &str, dtype: Dtype) -> Result<(Vec<usize>, Vec<u8>)> {
        let info = self.info(name)?;
        if info.dtype != dtype {
            return Err(format!("{name}: expected {dtype:?}, got {:?}", info.dtype).into());
        }
        let mut bytes = vec![0u8; info.data_offsets.1 - info.data_offsets.0];
        self.file
            .read_exact_at(&mut bytes, self.data_start + info.data_offsets.0 as u64)?;
        Ok((info.shape.clone(), bytes))
    }

    pub fn row(&self, name: &str, row: usize) -> Result<HostTensor> {
        let info = self.info(name)?;
        let (&rows, row_shape) = info
            .shape
            .split_first()
            .ok_or("scalar tensor has no rows")?;
        if row >= rows {
            return Err(format!("{name}: row {row} out of {rows}").into());
        }
        let width = row_shape.iter().product();
        Ok(HostTensor {
            shape: row_shape.to_vec(),
            data: self.read(info, row * width, width)?,
        })
    }

    fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.header
            .info(name)
            .ok_or_else(|| format!("unknown tensor {name}").into())
    }

    fn read(&self, info: &TensorInfo, first: usize, len: usize) -> Result<Vec<f32>> {
        let size = info.dtype.bitsize() / 8;
        let mut bytes = vec![0u8; len * size];
        let offset = self.data_start + (info.data_offsets.0 + first * size) as u64;
        self.file.read_exact_at(&mut bytes, offset)?;
        match info.dtype {
            Dtype::BF16 => Ok(bytes
                .chunks_exact(2)
                .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect()),
            Dtype::F32 => Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()),
            other => Err(format!("unsupported dtype {other:?}").into()),
        }
    }
}

pub struct Checkpoint {
    shard_of: HashMap<String, String>,
    shards: HashMap<String, SafetensorsFile>,
}

impl Checkpoint {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let index: serde_json::Value =
            serde_json::from_reader(File::open(dir.join("model.safetensors.index.json"))?)?;
        let shard_of: HashMap<String, String> = index["weight_map"]
            .as_object()
            .ok_or("index has no weight_map")?
            .iter()
            .filter_map(|(name, shard)| Some((name.clone(), shard.as_str()?.to_owned())))
            .collect();
        let shards = shard_of
            .values()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .filter(|shard| dir.join(shard).exists())
            .map(|shard| Ok((shard.clone(), SafetensorsFile::open(dir.join(shard))?)))
            .collect::<Result<_>>()?;
        Ok(Self { shard_of, shards })
    }

    pub fn tensor(&self, name: &str) -> Result<HostTensor> {
        self.shard(name)?.tensor(name)
    }

    pub fn tensor_bf16(&self, name: &str) -> Result<HostBf16Tensor> {
        self.shard(name)?.tensor_bf16(name)
    }

    pub fn tensor_fp8(&self, name: &str) -> Result<HostFp8Tensor> {
        self.shard(name)?.tensor_fp8(name)
    }

    pub fn tensor_nvfp4(&self, name: &str) -> Result<HostNvfp4Tensor> {
        self.shard(name)?.tensor_nvfp4(name)
    }

    pub fn scalar_f32(&self, name: &str) -> Result<f32> {
        self.shard(name)?.scalar_f32(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.shard_of.contains_key(name)
    }

    pub fn stacked_bf16(
        &self,
        first: &str,
        second: &str,
        rows: usize,
        cols: usize,
    ) -> Result<HostBf16Tensor> {
        let mut gate = self.tensor_bf16(first)?;
        let up = self.tensor_bf16(second)?;
        if gate.shape != [rows, cols] || up.shape != gate.shape {
            return Err("stacked matrix shape mismatch".into());
        }
        gate.data.extend(up.data);
        gate.shape = vec![2 * rows, cols];
        Ok(gate)
    }

    pub fn row(&self, name: &str, row: usize) -> Result<HostTensor> {
        self.shard(name)?.row(name, row)
    }

    pub fn matrix_bf16(
        &self,
        name: &str,
        shape: &[usize],
        stream: &Stream,
    ) -> Result<DenseMatrix<bf16>> {
        let tensor = self.tensor_bf16(name)?;
        if tensor.shape != shape || shape.len() < 2 {
            return Err(format!("{name}: expected shape {shape:?}, got {:?}", tensor.shape).into());
        }
        tensor.upload_matrix(stream)
    }

    pub fn vector(
        &self,
        name: &str,
        len: usize,
        offset: f32,
        stream: &Stream,
    ) -> Result<Tensor<f32>> {
        let tensor = self.tensor(name)?;
        if tensor.shape != [len] {
            return Err(format!("{name}: expected [{len}], got {:?}", tensor.shape).into());
        }
        Ok(api::copy_host_vec_to_device(&Arc::new(
            tensor.data.into_iter().map(|x| x + offset).collect(),
        ))
        .sync_on(stream)?)
    }

    pub fn embedding(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
        stream: &Stream,
    ) -> Result<Tensor<f32>> {
        let tensor = self.tensor(name)?;
        if tensor.shape != [rows, cols] {
            return Err(
                format!("{name}: expected [{rows}, {cols}], got {:?}", tensor.shape).into(),
            );
        }
        Ok(api::copy_host_vec_to_device(&Arc::new(tensor.data))
            .reshape(&[rows, cols])
            .sync_on(stream)?)
    }

    fn shard(&self, name: &str) -> Result<&SafetensorsFile> {
        let shard = self
            .shard_of
            .get(name)
            .ok_or_else(|| format!("unknown tensor {name}"))?;
        self.shards
            .get(shard)
            .ok_or_else(|| format!("{name} is in {shard}, which is not downloaded").into())
    }
}

#[cfg(test)]
mod dequant_tests {
    use super::*;

    fn fp8_to_f32(bits: u8) -> f32 {
        let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
        let exponent = (bits >> 3) & 0x0f;
        let fraction = bits & 0x07;
        if exponent == 0 {
            sign * (fraction as f32) * 2.0f32.powi(-9)
        } else {
            sign * (1.0 + (fraction as f32) / 8.0) * 2.0f32.powi(exponent as i32 - 7)
        }
    }

    #[test]
    #[ignore = "needs local official FP8 shards"]
    fn official_fp8_dequant_matches_python_bits() -> Result<()> {
        let model_dir = std::env::var_os("FIREPIT_FP8_MODEL")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("HOME")).join("Models/qwen3.6-35b-a3b-fp8"));
        let fixture = include_str!("../tests/fp8_dequant_samples.txt");
        let checkpoint = Checkpoint::open(&model_dir)?;
        for name in [
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.language_model.layers.0.mlp.experts.0.down_proj.weight",
        ] {
            let weights = checkpoint.tensor_fp8(name)?;
            let scales = checkpoint.tensor_bf16(&name.replace(".weight", ".weight_scale_inv"))?;
            let cols = weights.shape[1];
            let block_n = weights.shape[0] / scales.shape[0];
            let block_k = cols / scales.shape[1];
            for line in fixture.lines().filter(|line| line.starts_with(name)) {
                let values: Vec<&str> = line.split_whitespace().collect();
                let (row, col, python_bits) = (
                    values[1].parse::<usize>()?,
                    values[2].parse::<usize>()?,
                    values[3].parse::<u32>()?,
                );
                let weight = fp8_to_f32(weights.data[row * cols + col].0);
                let scale = scales.data[(row / block_n) * scales.shape[1] + col / block_k].to_f32();
                assert_eq!(
                    (weight * scale).to_bits(),
                    python_bits,
                    "{name} [{row}, {col}]"
                );
            }
        }
        Ok(())
    }

    fn e2m1(code: u8) -> f32 {
        let magnitude = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(code & 0x7) as usize];
        if code & 0x8 == 0 {
            magnitude
        } else {
            -magnitude
        }
    }

    #[test]
    #[ignore = "needs local ModelOpt NVFP4 shards"]
    fn modelopt_dequant_matches_python_bits() -> Result<()> {
        let model_dir = std::env::var_os("FIREPIT_NVFP4_MODEL")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("HOME")).join("Models/qwen3.6-35b-a3b-nvfp4"));
        let checkpoint = Checkpoint::open(&model_dir)?;
        let fixture = include_str!("../tests/modelopt_dequant_samples.txt");
        for line in fixture.lines() {
            let values: Vec<&str> = line.split_whitespace().collect();
            let (name, row, col, python_bits) = (
                values[0],
                values[1].parse::<usize>()?,
                values[2].parse::<usize>()?,
                values[3].parse::<u32>()?,
            );
            let companion = |suffix: &str| name.replace(".weight", suffix);
            let value = if checkpoint.contains(&companion(".weight_scale_2")) {
                let packed = checkpoint.tensor_nvfp4(name)?;
                let scales = checkpoint.tensor_fp8(&companion(".weight_scale"))?;
                let global = checkpoint.scalar_f32(&companion(".weight_scale_2"))?;
                let cols = packed.shape[1] * 2;
                let pair = packed.data[(row * cols + col) / 2];
                let code = if col % 2 == 0 {
                    pair.low()
                } else {
                    pair.high()
                };
                let scale = fp8_to_f32(scales.data[row * scales.shape[1] + col / 16].0);
                e2m1(code.0) * (scale * global)
            } else {
                let weights = checkpoint.tensor_fp8(name)?;
                let scale = checkpoint.scalar_f32(&companion(".weight_scale"))?;
                fp8_to_f32(weights.data[row * weights.shape[1] + col].0) * scale
            };
            assert_eq!(value.to_bits(), python_bits, "{name} [{row}, {col}]");
        }
        Ok(())
    }
}
