use crate::Result;
use serde_json::Value;
use std::fs::File;
use std::path::Path;
use tokenizers::Tokenizer;

pub fn eos_ids(model: &Path, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
    let generation = model.join("generation_config.json");
    let config = if generation.exists() {
        serde_json::from_reader::<_, Value>(File::open(generation)?)?
    } else {
        serde_json::from_reader::<_, Value>(File::open(model.join("config.json"))?)?
    };
    let ids = match &config["eos_token_id"] {
        Value::Number(id) => vec![id.as_u64().ok_or("invalid eos_token_id")?.try_into()?],
        Value::Array(ids) => ids
            .iter()
            .map(|id| Ok(id.as_u64().ok_or("invalid eos_token_id")?.try_into()?))
            .collect::<Result<Vec<_>>>()?,
        Value::Null => {
            let settings: Value =
                serde_json::from_reader(File::open(model.join("tokenizer_config.json"))?)?;
            let token = settings["eos_token"].as_str().ok_or("missing eos token")?;
            vec![tokenizer.token_to_id(token).ok_or("unknown eos token")?]
        }
        _ => return Err("invalid eos_token_id".into()),
    };
    if ids.is_empty() {
        return Err("empty eos_token_id".into());
    }
    Ok(ids)
}

pub fn greedy(logits: &[f32]) -> Result<u32> {
    let (id, _) = logits
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .ok_or("no finite logits")?;
    Ok(id.try_into()?)
}
