use crate::{Result, SafetensorsFile};
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::path::Path;

pub struct QualityChunk {
    pub tokens: Vec<u32>,
    pub top_ids: Vec<i32>,
    pub top_logp: Vec<f32>,
    pub tail_log_mass: Vec<f32>,
    pub chunk_size: usize,
    pub score_start: usize,
    pub top_k: usize,
    pub layers: usize,
}

impl QualityChunk {
    pub fn open(path: &Path) -> Result<Self> {
        let file = SafetensorsFile::open(path)?;
        if file.metadata("format") != Some("firepit-quality-v1") {
            return Err("unsupported quality reference format".into());
        }
        let number = |name: &str| -> Result<usize> {
            Ok(file
                .metadata(name)
                .ok_or(format!("missing {name}"))?
                .parse()?)
        };
        let chunk_size = number("chunk_size")?;
        let score_start = number("score_start")?;
        let top_k = number("top_k")?;
        let layers = number("layers")?;
        if chunk_size < 2
            || score_start == 0
            || score_start >= chunk_size
            || top_k == 0
            || layers == 0
        {
            return Err("invalid quality reference dimensions".into());
        }
        let tokens = file.tensor_i32("tokens")?;
        let top_ids = file.tensor_i32("top_ids")?;
        let top_logp = file.tensor("top_logp")?;
        let tail = file.tensor("tail_log_mass")?;
        let scored = chunk_size - score_start;
        if tokens.shape != [chunk_size]
            || top_ids.shape != [scored, top_k]
            || top_logp.shape != [scored, top_k]
            || tail.shape != [scored]
        {
            return Err("quality reference tensor shapes do not match metadata".into());
        }
        let tokens = tokens
            .data
            .into_iter()
            .map(|id| u32::try_from(id).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            tokens,
            top_ids: top_ids.data,
            top_logp: top_logp.data,
            tail_log_mass: tail.data,
            chunk_size,
            score_start,
            top_k,
            layers,
        })
    }

    pub fn row(&self, offset: usize) -> Result<(&[i32], &[f32], f32)> {
        if offset >= self.tail_log_mass.len() {
            return Err("quality row out of range".into());
        }
        let range = offset * self.top_k..(offset + 1) * self.top_k;
        Ok((
            &self.top_ids[range.clone()],
            &self.top_logp[range],
            self.tail_log_mass[offset],
        ))
    }

    pub fn scored_token(&self, offset: usize) -> Result<(usize, u32)> {
        let row = self
            .score_start
            .checked_add(offset)
            .ok_or("quality row overflow")?;
        Ok((
            row,
            *self.tokens.get(row).ok_or("quality row out of range")?,
        ))
    }
}

pub struct KlResult {
    pub kl: f64,
    pub same_top1: bool,
    pub tail_mass: f64,
}

pub fn truncated_kl(
    ids: &[i32],
    logp: &[f32],
    tail_log_mass: f32,
    logits: &[f32],
) -> Result<KlResult> {
    if ids.is_empty()
        || ids.len() != logp.len()
        || ids.len() >= logits.len()
        || !tail_log_mass.is_finite()
        || tail_log_mass > 0.0
        || logp.iter().any(|x| !x.is_finite() || *x > 0.0)
        || logits.iter().any(|x| !x.is_finite())
    {
        return Err("invalid KL input".into());
    }
    let mut seen = vec![false; logits.len()];
    for &id in ids {
        let index = usize::try_from(id)?;
        if index >= logits.len() || std::mem::replace(&mut seen[index], true) {
            return Err("invalid top-K ids".into());
        }
    }
    let (top1, &max_logit) = logits
        .iter()
        .enumerate()
        .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
        .ok_or("empty logits")?;
    let max_logit = max_logit as f64;
    let total: f64 = logits.iter().map(|&x| (x as f64 - max_logit).exp()).sum();
    let log_z = max_logit + total.ln();
    let top_q: f64 = ids
        .iter()
        .map(|&id| (logits[id as usize] as f64 - log_z).exp())
        .sum();
    let tail_q = if 1.0 - top_q > 1e-12 {
        1.0 - top_q
    } else {
        logits
            .iter()
            .enumerate()
            .filter(|(id, _)| !seen[*id])
            .map(|(_, &x)| (x as f64 - log_z).exp())
            .sum()
    };
    if tail_q <= 0.0 {
        return Err("firepit tail mass is zero".into());
    }
    let top_kl: f64 = ids
        .iter()
        .zip(logp)
        .map(|(&id, &p)| {
            let p = p as f64;
            p.exp() * (p - (logits[id as usize] as f64 - log_z))
        })
        .sum();
    let tail_log_mass = tail_log_mass as f64;
    let tail_mass = tail_log_mass.exp();
    Ok(KlResult {
        kl: top_kl + tail_mass * (tail_log_mass - tail_q.ln()),
        same_top1: top1 == ids[0] as usize,
        tail_mass,
    })
}

pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

pub fn t95(values: &[f64]) -> Result<Option<(f64, f64)>> {
    if values.len() < 2 {
        return Ok(None);
    }
    let center = mean(values);
    let variance =
        values.iter().map(|x| (x - center).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    let t = StudentsT::new(0.0, 1.0, (values.len() - 1) as f64)?;
    let half = t.inverse_cdf(0.975) * (variance / values.len() as f64).sqrt();
    Ok(Some((center, half)))
}

pub fn quantile(sorted: &[f64], p: f64) -> f64 {
    let index = (sorted.len() - 1) as f64 * p;
    let low = index.floor() as usize;
    let high = index.ceil() as usize;
    sorted[low] + (sorted[high] - sorted[low]) * index.fract()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn distribution_kl(p: &[f64], q: &[f64]) -> f64 {
        p.iter().zip(q).map(|(p, q)| p * (p / q).ln()).sum()
    }

    #[test]
    fn truncated_kl_matches_known_distribution_and_is_bounded() -> Result<()> {
        let p: [f64; 3] = [0.5, 0.3, 0.2];
        let q: [f64; 3] = [0.4, 0.4, 0.2];
        let logits: Vec<f32> = q.iter().map(|x| x.ln() as f32).collect();
        let result = truncated_kl(
            &[0],
            &[(p[0].ln() as f32)],
            (p[1] + p[2]).ln() as f32,
            &logits,
        )?;
        let coarse =
            p[0] * (p[0] / q[0]).ln() + (p[1] + p[2]) * ((p[1] + p[2]) / (q[1] + q[2])).ln();
        assert!((result.kl - coarse).abs() < 1e-6);
        assert!(result.kl <= distribution_kl(&p, &q) + 1e-6);
        assert!(result.same_top1);
        let identical = truncated_kl(
            &[0],
            &[(p[0].ln() as f32)],
            (p[1] + p[2]).ln() as f32,
            &p.iter().map(|x| x.ln() as f32).collect::<Vec<_>>(),
        )?;
        assert!(identical.kl.abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn scored_position_feeds_the_same_token_as_python_forward() -> Result<()> {
        let chunk = QualityChunk {
            tokens: vec![10, 11, 12, 13, 14, 15],
            top_ids: vec![],
            top_logp: vec![],
            tail_log_mass: vec![],
            chunk_size: 6,
            score_start: 3,
            top_k: 1,
            layers: 1,
        };
        assert_eq!(chunk.scored_token(0)?, (3, 13));
        assert_eq!(chunk.tokens[chunk.scored_token(0)?.0 + 1], 14);
        assert_eq!(chunk.scored_token(2)?, (5, 15));
        assert!(chunk.scored_token(3).is_err());
        Ok(())
    }

    #[test]
    fn two_chunk_interval_uses_student_t() -> Result<()> {
        let (center, half) = t95(&[0.0, 1.0])?.ok_or("missing interval")?;
        assert!((center - 0.5).abs() < 1e-12);
        assert!((half - 6.353102368).abs() < 1e-6);
        assert!(t95(&[0.5])?.is_none());
        Ok(())
    }
}
