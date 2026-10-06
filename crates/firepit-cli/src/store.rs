use crate::catalog::{Import, Recipe, Repo};
use crate::remote::{self, Remote, RemoteFile};
use firepit::{Dtype, Result, WeightFormat};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

const MANIFEST: &str = "firepit.json";
const INDEX: &str = "model.safetensors.index.json";

pub struct Store {
    root: PathBuf,
}

fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_reader(File::open(path)?)?)
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(value)? + "\n")?;
    Ok(())
}

fn wanted(file: &RemoteFile) -> bool {
    !file.name.starts_with('.') && !file.name.contains('/') && file.name != "README.md"
}

fn imported_file(import: &Import) -> String {
    format!("firepit-{}.safetensors", import.module)
}

fn repo_json(repo: &Repo) -> Value {
    json!({"repo": repo.id, "revision": repo.revision})
}

fn recipe_json(recipe: &Recipe) -> Value {
    json!({
        "model": recipe.model,
        "mode": recipe.mode,
        "base": repo_json(&recipe.base),
        "imports": recipe.imports.iter().map(|import| json!({
            "module": import.module,
            "from": repo_json(&import.from),
            "file": import.file,
        })).collect::<Vec<_>>(),
    })
}

fn tensor_header(name: &str, dtype: Dtype, shape: &[usize], bytes: u64) -> Result<Vec<u8>> {
    let mut header = json!({
        name: {"dtype": dtype, "shape": shape, "data_offsets": [0, bytes]},
    })
    .to_string()
    .into_bytes();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut framed = (header.len() as u64).to_le_bytes().to_vec();
    framed.extend(header);
    Ok(framed)
}

fn detach(dir: &Path, module: &str, file: &str) -> Result<Vec<&'static str>> {
    let owned = |name: &str| name == module || name.starts_with(&format!("{module}."));
    let mut index = read_json(&dir.join(INDEX))?;
    let map = index["weight_map"]
        .as_object_mut()
        .ok_or("index has no weight_map")?;
    map.retain(|name, _| !owned(name));
    map.insert(format!("{module}.weight"), file.into());
    write_json(&dir.join(INDEX), &index)?;
    let mut changed = vec![INDEX];

    let modelopt = dir.join("hf_quant_config.json");
    if modelopt.exists() {
        let mut config = read_json(&modelopt)?;
        if let Some(layers) = config["quantization"]["quantized_layers"].as_object_mut() {
            layers.remove(module);
        }
        write_json(&modelopt, &config)?;
        changed.push("hf_quant_config.json");
    }

    let mut config = read_json(&dir.join("config.json"))?;
    if let Some(quantization) = config.get_mut("quantization_config") {
        if let Some(layers) = quantization["quantized_layers"].as_object_mut() {
            layers.remove(module);
        }
        if let Some(groups) = quantization["config_groups"].as_object_mut() {
            for group in groups.values_mut() {
                if let Some(targets) = group["targets"].as_array_mut() {
                    targets.retain(|target| target != module);
                }
            }
        }
        write_json(&dir.join("config.json"), &config)?;
        changed.push("config.json");
    }
    Ok(changed)
}

fn verify(dir: &Path, recipe: &Recipe) -> Result<()> {
    let report = firepit::verify_checkpoint(dir)?;
    for import in recipe.imports {
        let name = format!("{}.weight", import.module);
        let weight = report
            .iter()
            .find(|weight| weight.name == name)
            .ok_or_else(|| format!("assembled checkpoint has no {name}"))?;
        if (weight.format, weight.dtype) != (WeightFormat::Bf16, Dtype::BF16) {
            return Err(format!(
                "{name} loads as {:?} {:?}, expected bf16",
                weight.format, weight.dtype
            )
            .into());
        }
    }
    Ok(())
}

impl Store {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn from_env() -> Result<Self> {
        let root = match std::env::var_os("FIREPIT_HOME") {
            Some(home) => PathBuf::from(home),
            None => PathBuf::from(
                std::env::var_os("HOME").ok_or("neither FIREPIT_HOME nor HOME is set")?,
            )
            .join(".cache/firepit"),
        };
        Ok(Self::new(root))
    }

    pub fn model_dir(&self, recipe: &Recipe) -> PathBuf {
        self.root
            .join("models")
            .join(recipe.model)
            .join(recipe.mode)
    }

    pub fn jit_dir(&self) -> PathBuf {
        self.root.join("jit")
    }

    fn staging_dir(&self, recipe: &Recipe) -> PathBuf {
        self.root
            .join("partial")
            .join(recipe.model)
            .join(recipe.mode)
    }

    pub fn installed(&self, recipe: &Recipe) -> Result<Option<PathBuf>> {
        let dir = self.model_dir(recipe);
        if !dir.exists() {
            return Ok(None);
        }
        let stale = |reason: &str| format!("{}: {reason}; remove it to pull again", dir.display());
        let manifest = read_json(&dir.join(MANIFEST)).map_err(|_| stale("no readable manifest"))?;
        if manifest["recipe"] != recipe_json(recipe) {
            return Err(stale("pulled from a different recipe").into());
        }
        for (name, entry) in manifest["files"]
            .as_object()
            .ok_or_else(|| stale("manifest lists no files"))?
        {
            let size = std::fs::metadata(dir.join(name))
                .map(|meta| meta.len())
                .ok();
            if size.is_none() || size != entry["size"].as_u64() {
                return Err(stale(&format!("{name} is missing or changed size")).into());
            }
        }
        Ok(Some(dir))
    }

    pub fn pull(&self, recipe: &Recipe, remote: &dyn Remote) -> Result<PathBuf> {
        if let Some(dir) = self.installed(recipe)? {
            return Ok(dir);
        }
        let staging = self.staging_dir(recipe);
        std::fs::create_dir_all(&staging)?;
        let mut files = BTreeMap::new();
        for file in remote.files(&recipe.base)?.into_iter().filter(wanted) {
            let path = staging.join(&file.name);
            if !path.exists() {
                eprintln!("fetching {} ({} bytes)", file.name, file.size);
                remote::resume(&path, file.size, &[], |offset, out| {
                    remote.read(&recipe.base, &file.name, offset..file.size, out)
                })?;
            }
            let digest = remote::sha256(&path)?;
            if file
                .sha256
                .as_ref()
                .is_some_and(|expected| *expected != digest)
            {
                std::fs::remove_file(&path)?;
                return Err(format!("{}: sha256 differs from the repository", file.name).into());
            }
            files.insert(file.name, (file.size, digest));
        }
        for import in recipe.imports {
            let name = format!("{}.weight", import.module);
            let file = imported_file(import);
            let path = staging.join(&file);
            if !path.exists() {
                let tensor = remote::locate(remote, &import.from, import.file, &name)?;
                let length = tensor.bytes.end - tensor.bytes.start;
                eprintln!("fetching {name} from {} ({length} bytes)", import.from.id);
                let header = tensor_header(&name, tensor.info.dtype, &tensor.info.shape, length)?;
                remote::resume(
                    &path,
                    header.len() as u64 + length,
                    &header,
                    |offset, out| {
                        let start = tensor.bytes.start + offset;
                        remote.read(&import.from, import.file, start..tensor.bytes.end, out)
                    },
                )?;
            }
            files.insert(
                file.clone(),
                (std::fs::metadata(&path)?.len(), remote::sha256(&path)?),
            );
            for changed in detach(&staging, import.module, &file)? {
                let path = staging.join(changed);
                files.insert(
                    changed.to_owned(),
                    (std::fs::metadata(&path)?.len(), remote::sha256(&path)?),
                );
            }
        }
        if let Err(error) = verify(&staging, recipe) {
            std::fs::remove_dir_all(&staging)?;
            return Err(format!(
                "assembled {}:{} failed verification: {error}",
                recipe.model, recipe.mode
            )
            .into());
        }
        write_json(
            &staging.join(MANIFEST),
            &json!({
                "recipe": recipe_json(recipe),
                "files": files.iter().map(|(name, (size, sha256))| {
                    (name.clone(), json!({"size": size, "sha256": sha256}))
                }).collect::<serde_json::Map<_, _>>(),
                "firepit": {"version": env!("CARGO_PKG_VERSION"), "commit": env!("FIREPIT_COMMIT")},
            }),
        )?;
        let dir = self.model_dir(recipe);
        std::fs::create_dir_all(dir.parent().ok_or("model directory has no parent")?)?;
        std::fs::rename(&staging, &dir)?;
        Ok(dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Measured;
    use safetensors::tensor::{SafeTensors, TensorView, serialize_to_file};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::ops::Range;

    const BASE: Repo = Repo {
        id: "test/base",
        revision: "1111111111111111111111111111111111111111",
    };
    const SOURCE: Repo = Repo {
        id: "test/source",
        revision: "2222222222222222222222222222222222222222",
    };
    const MEASURED: Measured = Measured {
        decode_ms: 1.0,
        delta_nll: 0.0,
        delta_nll_ci: 0.0,
        kl: 0.0,
        top1: 1.0,
        at: "test",
    };
    static RECIPE: Recipe = Recipe {
        model: "small",
        mode: "mixed",
        default: true,
        summary: "",
        size_gb: 0.0,
        base: BASE,
        imports: &[Import {
            module: "lm_head",
            from: SOURCE,
            file: "source.safetensors",
        }],
        measured: MEASURED,
    };
    static MISSING_IMPORT: Recipe = Recipe {
        model: "small",
        mode: "missing",
        default: false,
        summary: "",
        size_gb: 0.0,
        base: BASE,
        imports: &[Import {
            module: "missing",
            from: SOURCE,
            file: "source.safetensors",
        }],
        measured: MEASURED,
    };
    const NORM: &str = "model.language_model.norm.weight";

    struct LocalRemote {
        root: PathBuf,
        offline: bool,
    }

    impl LocalRemote {
        fn repo(&self, repo: &Repo) -> PathBuf {
            self.root.join(repo.id.replace('/', "_"))
        }
    }

    impl Remote for LocalRemote {
        fn files(&self, repo: &Repo) -> Result<Vec<RemoteFile>> {
            if self.offline {
                return Err("offline".into());
            }
            std::fs::read_dir(self.repo(repo))?
                .map(|entry| {
                    let path = entry?.path();
                    let name = path.file_name().unwrap().to_string_lossy().into_owned();
                    Ok(RemoteFile {
                        sha256: name
                            .ends_with(".safetensors")
                            .then(|| remote::sha256(&path))
                            .transpose()?,
                        size: std::fs::metadata(&path)?.len(),
                        name,
                    })
                })
                .collect()
        }

        fn read(
            &self,
            repo: &Repo,
            file: &str,
            range: Range<u64>,
            out: &mut dyn Write,
        ) -> Result<()> {
            if self.offline {
                return Err("offline".into());
            }
            let mut source = File::open(self.repo(repo).join(file))?;
            source.seek(SeekFrom::Start(range.start))?;
            let copied = std::io::copy(&mut source.take(range.end - range.start), out)?;
            if copied != range.end - range.start {
                return Err("short read".into());
            }
            Ok(())
        }
    }

    fn small_config() -> Value {
        serde_json::from_str(include_str!("../../firepit/tests/fixtures/small-qwen.json")).unwrap()
    }

    fn dims(config: &Value) -> (usize, usize) {
        let text = &config["text_config"];
        let size = |name: &str| text[name].as_u64().unwrap() as usize;
        (size("vocab_size"), size("hidden_size"))
    }

    fn write_shard(
        dir: &Path,
        file: &str,
        tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)],
    ) -> Result<()> {
        let views = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                Ok((*name, TensorView::new(*dtype, shape.clone(), data)?))
            })
            .collect::<Result<Vec<_>>>()?;
        serialize_to_file(views, None, &dir.join(file))?;
        let index = json!({"weight_map": tensors.iter().map(|t| (t.0.to_owned(), json!(file))).collect::<serde_json::Map<_, _>>()});
        write_json(&dir.join(INDEX), &index)
    }

    fn source_lm_head(vocab: usize, hidden: usize) -> Vec<u8> {
        (0..vocab * hidden)
            .flat_map(|i| half::bf16::from_f32(i as f32 / 64.0).to_le_bytes())
            .collect()
    }

    fn remote_with(root: &Path, stray_scale: bool) -> Result<LocalRemote> {
        let config = small_config();
        let (vocab, hidden) = dims(&config);
        let base = root.join("test_base");
        std::fs::create_dir_all(&base)?;
        let mut quantized = config.clone();
        quantized["quantization_config"] = json!({
            "quant_method": "modelopt",
            "quantized_layers": {"lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 16}},
            "config_groups": {"group_1": {"targets": ["lm_head", "model.language_model.layers.0.mlp.experts"]}},
        });
        write_json(&base.join("config.json"), &quantized)?;
        write_json(
            &base.join("hf_quant_config.json"),
            &json!({
                "producer": {"name": "modelopt"},
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": {"lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 16}},
                },
            }),
        )?;
        std::fs::write(base.join("README.md"), "skip me")?;
        std::fs::write(base.join(".gitattributes"), "skip me")?;
        let mut tensors = vec![
            (
                "lm_head.weight",
                Dtype::U8,
                vec![vocab, hidden / 2],
                vec![0x22; vocab * hidden / 2],
            ),
            (
                "lm_head.weight_scale",
                Dtype::F8_E4M3,
                vec![vocab, hidden / 16],
                vec![0x38; vocab * hidden / 16],
            ),
            (
                "lm_head.weight_scale_2",
                Dtype::F32,
                vec![],
                1.0f32.to_le_bytes().to_vec(),
            ),
            (
                "lm_head.input_scale",
                Dtype::F32,
                vec![],
                1.0f32.to_le_bytes().to_vec(),
            ),
            (NORM, Dtype::BF16, vec![hidden], vec![0; 2 * hidden]),
        ];
        if stray_scale {
            tensors.push((
                "model.language_model.norm.weight_scale",
                Dtype::F32,
                vec![],
                vec![0; 4],
            ));
        }
        write_shard(&base, "model.safetensors", &tensors)?;

        let source = root.join("test_source");
        std::fs::create_dir_all(&source)?;
        write_shard(
            &source,
            "source.safetensors",
            &[
                (NORM, Dtype::BF16, vec![hidden], vec![1; 2 * hidden]),
                (
                    "lm_head.weight",
                    Dtype::BF16,
                    vec![vocab, hidden],
                    source_lm_head(vocab, hidden),
                ),
            ],
        )?;
        Ok(LocalRemote {
            root: root.to_owned(),
            offline: false,
        })
    }

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("firepit-cli-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn pull_assembles_the_imported_lm_head() -> Result<()> {
        let root = temp("assemble");
        let remote = remote_with(&root.join("remote"), false)?;
        let store = Store::new(root.join("home"));
        let dir = store.pull(&RECIPE, &remote)?;
        assert_eq!(dir, root.join("home/models/small/mixed"));
        assert!(!root.join("home/partial/small/mixed").exists());
        assert!(!dir.join("README.md").exists() && !dir.join(".gitattributes").exists());

        let index = read_json(&dir.join(INDEX))?;
        let map = index["weight_map"].as_object().unwrap();
        assert_eq!(map["lm_head.weight"], "firepit-lm_head.safetensors");
        assert!(
            map.keys()
                .all(|name| name == "lm_head.weight" || !name.starts_with("lm_head."))
        );
        assert_eq!(map[NORM], "model.safetensors");
        let modelopt = read_json(&dir.join("hf_quant_config.json"))?;
        assert!(
            modelopt["quantization"]["quantized_layers"]
                .get("lm_head")
                .is_none()
        );
        let config = read_json(&dir.join("config.json"))?;
        assert!(
            config["quantization_config"]["quantized_layers"]
                .get("lm_head")
                .is_none()
        );
        assert_eq!(
            config["quantization_config"]["config_groups"]["group_1"]["targets"],
            json!(["model.language_model.layers.0.mlp.experts"])
        );

        let (vocab, hidden) = dims(&small_config());
        let bytes = std::fs::read(dir.join("firepit-lm_head.safetensors"))?;
        let imported = SafeTensors::deserialize(&bytes)?.tensor("lm_head.weight")?;
        assert_eq!(
            (imported.dtype(), imported.shape()),
            (Dtype::BF16, &[vocab, hidden][..])
        );
        assert_eq!(imported.data(), source_lm_head(vocab, hidden).as_slice());

        let manifest = read_json(&dir.join(MANIFEST))?;
        assert_eq!(manifest["recipe"], recipe_json(&RECIPE));
        for (name, entry) in manifest["files"].as_object().unwrap() {
            assert_eq!(entry["sha256"], remote::sha256(&dir.join(name))?, "{name}");
        }
        let report = firepit::verify_checkpoint(&dir)?;
        let lm_head = report.iter().find(|w| w.name == "lm_head.weight").unwrap();
        assert_eq!(lm_head.format, WeightFormat::Bf16);
        std::fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn pull_skips_an_installed_model_and_flags_changed_files() -> Result<()> {
        let root = temp("skip");
        let remote = remote_with(&root.join("remote"), false)?;
        let store = Store::new(root.join("home"));
        let dir = store.pull(&RECIPE, &remote)?;
        let offline = LocalRemote {
            root: root.join("remote"),
            offline: true,
        };
        assert_eq!(store.pull(&RECIPE, &offline)?, dir);
        std::fs::write(dir.join("config.json"), "{}")?;
        assert!(store.installed(&RECIPE).is_err());
        assert!(store.pull(&RECIPE, &offline).is_err());
        std::fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn failed_pulls_leave_no_model() -> Result<()> {
        let root = temp("fail");
        let store = Store::new(root.join("home"));
        let remote = remote_with(&root.join("remote"), true)?;
        assert!(store.pull(&RECIPE, &remote).is_err());
        assert!(!store.model_dir(&RECIPE).exists());
        assert!(!root.join("home/partial/small/mixed").exists());

        let remote = remote_with(&root.join("clean"), false)?;
        assert!(store.pull(&MISSING_IMPORT, &remote).is_err());
        assert!(!store.model_dir(&MISSING_IMPORT).exists());
        std::fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn located_ranges_hold_the_tensor_bytes() -> Result<()> {
        let root = temp("locate");
        let remote = remote_with(&root, false)?;
        let (vocab, hidden) = dims(&small_config());
        let tensor = remote::locate(&remote, &SOURCE, "source.safetensors", "lm_head.weight")?;
        assert_eq!(tensor.info.shape, vec![vocab, hidden]);
        let mut bytes = Vec::new();
        remote.read(&SOURCE, "source.safetensors", tensor.bytes, &mut bytes)?;
        assert_eq!(bytes, source_lm_head(vocab, hidden));
        assert!(remote::locate(&remote, &SOURCE, "source.safetensors", "missing.weight").is_err());
        std::fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn resume_appends_to_a_matching_partial_file() -> Result<()> {
        let root = temp("resume");
        std::fs::create_dir_all(&root)?;
        let path = root.join("file.bin");
        std::fs::write(root.join("file.bin.part"), b"HDRabc")?;
        remote::resume(&path, 9, b"HDR", |offset, out| {
            assert_eq!(offset, 3);
            Ok(out.write_all(b"def")?)
        })?;
        assert_eq!(std::fs::read(&path)?, b"HDRabcdef");
        std::fs::write(root.join("other.bin.part"), b"XXXabc")?;
        remote::resume(&root.join("other.bin"), 9, b"HDR", |offset, out| {
            assert_eq!(offset, 0);
            Ok(out.write_all(b"abcdef")?)
        })?;
        assert_eq!(std::fs::read(root.join("other.bin"))?, b"HDRabcdef");
        assert!(
            remote::resume(&root.join("short.bin"), 9, b"", |_, out| Ok(
                out.write_all(b"ab")?
            ))
            .is_err()
        );
        std::fs::remove_dir_all(&root)?;
        Ok(())
    }
}
