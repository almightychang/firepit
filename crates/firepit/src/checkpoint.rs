use crate::Result;
use half::bf16;
use safetensors::Dtype;
use safetensors::tensor::{Metadata, TensorInfo};
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

pub struct HostTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
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
        self.header.metadata().as_ref()?.get(key).map(String::as_str)
    }

    pub fn tensor(&self, name: &str) -> Result<HostTensor> {
        let info = self.info(name)?;
        let len = info.shape.iter().product();
        Ok(HostTensor {
            shape: info.shape.clone(),
            data: self.read(info, 0, len)?,
        })
    }

    pub fn row(&self, name: &str, row: usize) -> Result<HostTensor> {
        let info = self.info(name)?;
        let (&rows, row_shape) = info.shape.split_first().ok_or("scalar tensor has no rows")?;
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

    pub fn row(&self, name: &str, row: usize) -> Result<HostTensor> {
        self.shard(name)?.row(name, row)
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
