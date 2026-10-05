use crate::catalog::Repo;
use firepit::Result;
use safetensors::tensor::{Metadata, TensorInfo};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::ops::Range;
use std::path::Path;

const MAX_HEADER_BYTES: u64 = 100 << 20;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteFile {
    pub name: String,
    pub size: u64,
    pub sha256: Option<String>,
}

pub trait Remote {
    fn files(&self, repo: &Repo) -> Result<Vec<RemoteFile>>;
    fn read(&self, repo: &Repo, file: &str, range: Range<u64>, out: &mut dyn Write) -> Result<()>;
}

pub struct HuggingFace {
    agent: ureq::Agent,
}

impl HuggingFace {
    const ENDPOINT: &str = "https://huggingface.co";

    pub fn new() -> Self {
        Self {
            agent: ureq::Agent::new_with_defaults(),
        }
    }
}

impl Remote for HuggingFace {
    fn files(&self, repo: &Repo) -> Result<Vec<RemoteFile>> {
        let url = format!(
            "{}/api/models/{}/revision/{}?blobs=true",
            Self::ENDPOINT,
            repo.id,
            repo.revision
        );
        let listing: Value =
            serde_json::from_reader(self.agent.get(&url).call()?.into_body().into_reader())?;
        if listing["sha"] != repo.revision {
            return Err(format!("{}: listing is not revision {}", repo.id, repo.revision).into());
        }
        listing["siblings"]
            .as_array()
            .ok_or_else(|| format!("{}: listing has no files", repo.id))?
            .iter()
            .map(|file| {
                Ok(RemoteFile {
                    name: file["rfilename"]
                        .as_str()
                        .ok_or("listing file has no name")?
                        .to_owned(),
                    size: file["size"].as_u64().ok_or("listing file has no size")?,
                    sha256: file["lfs"]["sha256"].as_str().map(str::to_owned),
                })
            })
            .collect()
    }

    fn read(&self, repo: &Repo, file: &str, range: Range<u64>, out: &mut dyn Write) -> Result<()> {
        if range.is_empty() {
            return Ok(());
        }
        let url = format!(
            "{}/{}/resolve/{}/{file}",
            Self::ENDPOINT,
            repo.id,
            repo.revision
        );
        let response = self
            .agent
            .get(&url)
            .header("Range", format!("bytes={}-{}", range.start, range.end - 1))
            .call()?;
        let length = response
            .headers()
            .get("content-length")
            .and_then(|value| value.to_str().ok()?.parse::<u64>().ok());
        if response.status() != 206 || length != Some(range.end - range.start) {
            return Err(format!(
                "{file}: expected {} bytes of a partial response, got status {} length {length:?}",
                range.end - range.start,
                response.status()
            )
            .into());
        }
        let copied = io::copy(&mut response.into_body().into_reader(), out)?;
        if copied != range.end - range.start {
            return Err(format!("{file}: response ended after {copied} bytes").into());
        }
        Ok(())
    }
}

pub struct RemoteTensor {
    pub info: TensorInfo,
    pub bytes: Range<u64>,
}

pub fn locate(remote: &dyn Remote, repo: &Repo, file: &str, name: &str) -> Result<RemoteTensor> {
    let mut prefix = Vec::with_capacity(8);
    remote.read(repo, file, 0..8, &mut prefix)?;
    let header_len = u64::from_le_bytes(prefix.as_slice().try_into()?);
    if header_len > MAX_HEADER_BYTES {
        return Err(format!("{file}: header of {header_len} bytes is too large").into());
    }
    let mut header = Vec::with_capacity(header_len as usize);
    remote.read(repo, file, 8..8 + header_len, &mut header)?;
    let metadata: Metadata = serde_json::from_slice(&header)?;
    let info = metadata
        .info(name)
        .ok_or_else(|| format!("{file} has no tensor {name}"))?
        .clone();
    let (start, end) = info.data_offsets;
    let elements: usize = info.shape.iter().product();
    if end < start || elements * info.dtype.bitsize() != (end - start) * 8 {
        return Err(format!("{file}: {name} byte range disagrees with its dtype and shape").into());
    }
    let data = 8 + header_len;
    Ok(RemoteTensor {
        info,
        bytes: data + start as u64..data + end as u64,
    })
}

pub fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

pub fn resume(
    path: &Path,
    expected: u64,
    prefix: &[u8],
    fetch: impl FnOnce(u64, &mut File) -> Result<()>,
) -> Result<()> {
    let part = path.with_file_name(format!(
        "{}.part",
        path.file_name()
            .ok_or("download has no file name")?
            .to_string_lossy()
    ));
    let kept = match std::fs::metadata(&part) {
        Ok(meta)
            if (prefix.len() as u64..=expected).contains(&meta.len())
                && starts_with(&part, prefix)? =>
        {
            meta.len()
        }
        _ => {
            File::create(&part)?.write_all(prefix)?;
            prefix.len() as u64
        }
    };
    let mut file = OpenOptions::new().append(true).open(&part)?;
    fetch(kept - prefix.len() as u64, &mut file)?;
    file.sync_all()?;
    let written = file.metadata()?.len();
    if written != expected {
        return Err(format!("{}: wrote {written} of {expected} bytes", path.display()).into());
    }
    std::fs::rename(&part, path)?;
    Ok(())
}

fn starts_with(path: &Path, prefix: &[u8]) -> Result<bool> {
    let mut start = vec![0u8; prefix.len()];
    File::open(path)?.read_exact(&mut start)?;
    Ok(start == prefix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::RECIPES;

    #[test]
    #[ignore = "reads a few kilobytes from huggingface.co"]
    fn hugging_face_serves_listings_and_tensor_ranges() -> Result<()> {
        let import = RECIPES
            .iter()
            .flat_map(|recipe| recipe.imports)
            .find(|import| import.module == "lm_head")
            .ok_or("catalog imports no lm_head")?;
        let hub = HuggingFace::new();
        let listing = hub.files(&import.from)?;
        let file = listing
            .iter()
            .find(|file| file.name == import.file)
            .ok_or("import file is not listed")?;
        assert!(file.sha256.is_some());
        let tensor = locate(&hub, &import.from, import.file, "lm_head.weight")?;
        assert_eq!(tensor.info.dtype, safetensors::Dtype::BF16);
        assert!(tensor.bytes.end <= file.size);
        println!("{import:?}: {:?} {:?}", tensor.info.shape, tensor.bytes);
        Ok(())
    }
}
