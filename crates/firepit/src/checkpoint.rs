use crate::Result;
use cuda_async::device_operation::DeviceOp;
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
        let info = self.info(name)?;
        if info.dtype != Dtype::I32 {
            return Err(format!("{name}: expected I32, got {:?}", info.dtype).into());
        }
        let mut bytes = vec![0u8; info.data_offsets.1 - info.data_offsets.0];
        self.file
            .read_exact_at(&mut bytes, self.data_start + info.data_offsets.0 as u64)?;
        Ok(HostI32Tensor {
            shape: info.shape.clone(),
            data: bytes
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        })
    }

    pub fn tensor_bf16(&self, name: &str) -> Result<HostBf16Tensor> {
        let info = self.info(name)?;
        if info.dtype != Dtype::BF16 {
            return Err(format!("{name}: expected BF16, got {:?}", info.dtype).into());
        }
        let mut bytes = vec![0u8; info.data_offsets.1 - info.data_offsets.0];
        self.file
            .read_exact_at(&mut bytes, self.data_start + info.data_offsets.0 as u64)?;
        Ok(HostBf16Tensor {
            shape: info.shape.clone(),
            data: bytes
                .chunks_exact(2)
                .map(|b| bf16::from_le_bytes([b[0], b[1]]))
                .collect(),
        })
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
