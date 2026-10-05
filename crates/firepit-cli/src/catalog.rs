use std::fmt;
use std::str::FromStr;

#[derive(Debug, PartialEq, Eq)]
pub struct Repo {
    pub id: &'static str,
    pub revision: &'static str,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Import {
    pub module: &'static str,
    pub from: Repo,
    pub file: &'static str,
}

#[derive(Debug)]
pub struct Measured {
    pub decode_ms: f64,
    pub delta_nll: f64,
    pub delta_nll_ci: f64,
    pub kl: f64,
    pub top1: f64,
    pub at: &'static str,
}

#[derive(Debug)]
pub struct Recipe {
    pub model: &'static str,
    pub mode: &'static str,
    pub default: bool,
    pub summary: &'static str,
    pub size_gb: f64,
    pub base: Repo,
    pub imports: &'static [Import],
    pub measured: Measured,
}

const QWEN36_BF16: Repo = Repo {
    id: "Qwen/Qwen3.6-35B-A3B",
    revision: "995ad96eacd98c81ed38be0c5b274b04031597b0",
};

pub const MEASUREMENTS: &str = "docs/research/measurements/nvfp4-gb10.md";

pub const RECIPES: &[Recipe] = &[
    Recipe {
        model: "qwen3.6",
        mode: "nvfp4",
        default: true,
        summary: "NVIDIA NVFP4 experts, FP8 projections, bf16 lm_head",
        size_gb: 24.4,
        base: Repo {
            id: "nvidia/Qwen3.6-35B-A3B-NVFP4",
            revision: "1355db6a052410cfd62085d94b58866fd0f2c3c5",
        },
        imports: &[Import {
            module: "lm_head",
            from: QWEN36_BF16,
            file: "model-00026-of-00026.safetensors",
        }],
        measured: Measured {
            decode_ms: 22.9,
            delta_nll: 1.08e-2,
            delta_nll_ci: 2.2e-3,
            kl: 2.53e-2,
            top1: 0.9333,
            at: "abd177f",
        },
    },
    Recipe {
        model: "qwen3.6",
        mode: "fp8",
        default: false,
        summary: "Qwen FP8, 128x128 block scales",
        size_gb: 35.0,
        base: Repo {
            id: "Qwen/Qwen3.6-35B-A3B-FP8",
            revision: "95a723d08a9490559dae23d0cff1d9466213d989",
        },
        imports: &[],
        measured: Measured {
            decode_ms: 24.3,
            delta_nll: 3.3e-3,
            delta_nll_ci: 1.1e-3,
            kl: 9.3e-3,
            top1: 0.959,
            at: "88c3465",
        },
    },
    Recipe {
        model: "qwen3.6",
        mode: "bf16",
        default: false,
        summary: "Original bf16 weights",
        size_gb: 67.0,
        base: QWEN36_BF16,
        imports: &[],
        measured: Measured {
            decode_ms: 27.7,
            delta_nll: 0.0,
            delta_nll_ci: 0.0,
            kl: 4.8e-3,
            top1: 0.971,
            at: "88c3465",
        },
    },
];

#[derive(Clone, Copy, Debug)]
pub struct ModelRef(pub &'static Recipe);

impl ModelRef {
    pub fn recipe(self) -> &'static Recipe {
        self.0
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.0.model, self.0.mode)
    }
}

fn supported() -> String {
    RECIPES
        .iter()
        .map(|recipe| {
            let default = if recipe.default { " (default)" } else { "" };
            format!("{}:{}{default}", recipe.model, recipe.mode)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

impl FromStr for ModelRef {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, String> {
        let (model, mode) = match value.split_once(':') {
            Some((model, mode)) => (model, Some(mode)),
            None => (value, None),
        };
        RECIPES
            .iter()
            .find(|recipe| {
                recipe.model == model && mode.map_or(recipe.default, |mode| recipe.mode == mode)
            })
            .map(ModelRef)
            .ok_or_else(|| format!("unknown model {value}; supported: {}", supported()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_refs_resolve_to_recipes() {
        for (name, mode) in [
            ("qwen3.6", "nvfp4"),
            ("qwen3.6:nvfp4", "nvfp4"),
            ("qwen3.6:fp8", "fp8"),
            ("qwen3.6:bf16", "bf16"),
        ] {
            let model: ModelRef = name.parse().unwrap();
            assert_eq!(model.recipe().mode, mode, "{name}");
        }
        for name in ["qwen3.5", "qwen3.6:int4", "qwen3.6:", ":fp8", ""] {
            let error = name.parse::<ModelRef>().unwrap_err();
            assert!(error.contains("qwen3.6:nvfp4 (default)"), "{error}");
        }
    }

    #[test]
    fn catalog_has_one_default_per_model_and_pinned_revisions() {
        for recipe in RECIPES {
            let defaults = RECIPES
                .iter()
                .filter(|other| other.model == recipe.model && other.default)
                .count();
            assert_eq!(defaults, 1, "{}", recipe.model);
            let repos = std::iter::once(&recipe.base).chain(recipe.imports.iter().map(|i| &i.from));
            for repo in repos {
                assert_eq!(repo.revision.len(), 40, "{}", repo.id);
                assert!(repo.revision.bytes().all(|b| b.is_ascii_hexdigit()));
            }
        }
    }
}
