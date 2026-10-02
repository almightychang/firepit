mod common;

use common::{check, strategy};
use cuda_async::device_operation::DeviceOp;
use cuda_core::Stream;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, ToHostVec};
use cutile::tile_kernel::TileKernel;
use firepit_kernels::attention::{AttentionDims, cache_append, online_attention, output_gate};
use proptest::prelude::*;
use std::sync::Arc;

fn gpu_attention(
    stream: &Arc<Stream>,
    dims: &AttentionDims,
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    len: usize,
    scale: f32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let d = dims.head_dim();
    let kv = dims.kv_heads();
    let x = api::copy_host_vec_to_device(&Arc::new(q.to_vec()))
        .reshape(&[dims.q_heads(), d])
        .sync_on(stream)?;
    let mut padded_k = keys.to_vec();
    let mut padded_v = values.to_vec();
    padded_k.resize(dims.max_seq() * kv * d, 0.0);
    padded_v.resize(dims.max_seq() * kv * d, 0.0);
    let k = api::copy_host_vec_to_device(&Arc::new(padded_k))
        .reshape(&[dims.max_seq(), kv, d])
        .sync_on(stream)?;
    let v = api::copy_host_vec_to_device(&Arc::new(padded_v))
        .reshape(&[dims.max_seq(), kv, d])
        .sync_on(stream)?;
    let pos = api::copy_host_vec_to_device(&Arc::new(vec![(len - 1) as i32])).sync_on(stream)?;
    let mut out = api::zeros::<f32>(&[dims.q_heads(), d]).sync_on(stream)?;
    online_attention((&mut out).partition([1, d]), &x, &k, &v, &pos, scale)
        .generics(dims.attention_generics())
        .sync_on(stream)?;
    Ok(out.to_host_vec().sync_on(stream)?)
}

fn oracle(
    dims: &AttentionDims,
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    len: usize,
    scale: f32,
) -> Vec<f64> {
    let d = dims.head_dim();
    let kv = dims.kv_heads();
    let group = dims.q_heads() / kv;
    let mut out = vec![0.0f64; dims.q_heads() * d];
    for h in 0..dims.q_heads() {
        let kh = h / group;
        let scores: Vec<f64> = (0..len)
            .map(|t| {
                let base = (t * kv + kh) * d;
                (0..d)
                    .map(|i| q[h * d + i] as f64 * keys[base + i] as f64)
                    .sum::<f64>()
                    * scale as f64
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|&s| (s - max).exp()).collect();
        let denom: f64 = weights.iter().sum();
        for t in 0..len {
            let base = (t * kv + kh) * d;
            for i in 0..d {
                out[h * d + i] += weights[t] * values[base + i] as f64 / denom;
            }
        }
    }
    out
}

// Score error E (Higham–Mary on q·k) perturbs softmax weights by at most (e^{2E} − 1)·a;
// the online rescaling adds about two roundings per position to the weighted sum;
// weights below the f32 normal range flush to zero, which bounds the absolute floor.
fn allowance(
    dims: &AttentionDims,
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    len: usize,
    scale: f32,
) -> Vec<f64> {
    let u = f32::EPSILON as f64 / 2.0;
    let d = dims.head_dim();
    let kv = dims.kv_heads();
    let group = dims.q_heads() / kv;
    let mut out = vec![0.0f64; dims.q_heads() * d];
    for h in 0..dims.q_heads() {
        let kh = h / group;
        let row = |t: usize| (t * kv + kh) * d;
        let dot = |t: usize, f: fn(f64) -> f64| -> f64 {
            (0..d)
                .map(|i| f(q[h * d + i] as f64 * keys[row(t) + i] as f64))
                .sum::<f64>()
                * scale as f64
        };
        let scores: Vec<f64> = (0..len).map(|t| dot(t, |x| x)).collect();
        let score_error = (0..len)
            .map(|t| check::gamma_tilde(d) * dot(t, f64::abs) + 4.0 * u)
            .fold(0.0, f64::max);
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|&s| (s - max).exp()).collect();
        let denom: f64 = weights.iter().sum();
        let factor = check::gamma_tilde(2 * len + 2) + (2.0 * score_error).exp_m1() + 8.0 * u;
        let largest_value = (0..len)
            .flat_map(|t| values[row(t)..row(t) + d].iter())
            .fold(0.0f64, |m, &v| m.max((v as f64).abs()));
        let underflow = len as f64 * f32::MIN_POSITIVE as f64 * largest_value;
        for i in 0..d {
            out[h * d + i] = underflow;
        }
        for t in 0..len {
            for i in 0..d {
                out[h * d + i] += factor * weights[t] / denom * (values[row(t) + i] as f64).abs();
            }
        }
    }
    out
}

fn gpu_gate(
    stream: &Arc<Stream>,
    values: &[f32],
    gate: &[f32],
    d: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let rows = values.len() / d;
    let x = api::copy_host_vec_to_device(&Arc::new(values.to_vec()))
        .reshape(&[rows, d])
        .sync_on(stream)?;
    let g = api::copy_host_vec_to_device(&Arc::new(gate.to_vec()))
        .reshape(&[rows, d])
        .sync_on(stream)?;
    let mut out = api::zeros::<f32>(&[rows, d]).sync_on(stream)?;
    output_gate((&mut out).partition([1, d]), &x, &g)
        .generics(vec![d.to_string()])
        .sync_on(stream)?;
    Ok(out.to_host_vec().sync_on(stream)?)
}

#[test]
fn attention_matches_f64_oracle() {
    let stream = common::stream();
    common::runner(64, file!())
        .run(
            &(
                prop::sample::select(vec![(1usize, 1usize), (2, 1), (4, 2), (4, 4)]),
                prop::sample::select(vec![32usize, 64]),
                prop::sample::select(vec![1usize, 2, 7, 64, 256]),
            )
                .prop_flat_map(|((qh, kvh), d, len)| {
                    (
                        Just((qh, kvh, d, len)),
                        strategy::values(qh * d),
                        strategy::values(len * kvh * d),
                        strategy::values(len * kvh * d),
                    )
                }),
            |((qh, kvh, d, len), q, keys, values)| {
                let dims = AttentionDims::new(256, qh, kvh, d).unwrap();
                let actual = gpu_attention(&stream, &dims, &q, &keys, &values, len, dims.scale())
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let expected = oracle(&dims, &q, &keys, &values, len, dims.scale());
                let allowed = allowance(&dims, &q, &keys, &values, len, dims.scale());
                let excess = actual
                    .iter()
                    .zip(&expected)
                    .zip(&allowed)
                    .map(|((&a, &e), &b)| (a as f64 - e).abs() / b.max(f64::MIN_POSITIVE))
                    .fold(0.0, f64::max);
                prop_assert!(excess <= 1.0, "excess {excess}");
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn attention_preserves_permutation_constant_values_and_score_shift()
-> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let dims = AttentionDims::new(16, 1, 1, 32)?;
    let q: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect();
    let keys: Vec<f32> = (0..7 * 32)
        .map(|i| ((i * 19 % 71) as f32 - 35.0) / 8.0)
        .collect();
    let values: Vec<f32> = (0..7 * 32)
        .map(|i| ((i * 11 % 67) as f32 - 33.0) / 9.0)
        .collect();
    let base = gpu_attention(&stream, &dims, &q, &keys, &values, 7, dims.scale())?;
    let perm = [4, 2, 6, 0, 5, 1, 3];
    let reordered_k: Vec<f32> = perm
        .iter()
        .flat_map(|&t| keys[t * 32..(t + 1) * 32].iter().copied())
        .collect();
    let reordered_v: Vec<f32> = perm
        .iter()
        .flat_map(|&t| values[t * 32..(t + 1) * 32].iter().copied())
        .collect();
    let reordered = gpu_attention(
        &stream,
        &dims,
        &q,
        &reordered_k,
        &reordered_v,
        7,
        dims.scale(),
    )?;
    assert!(check::max_abs(&base, &reordered) < 1e-5);
    let same = vec![3.25f32; 7 * 32];
    let equal = gpu_attention(&stream, &dims, &q, &keys, &same, 7, dims.scale())?;
    assert!(equal.iter().all(|&x| (x - 3.25).abs() < 1e-5));
    let shifted: Vec<f32> = keys
        .chunks_exact(32)
        .flat_map(|row| row.iter().zip(&q).map(|(&k, &q)| k + 5.0 * q))
        .collect();
    let shifted_out = gpu_attention(&stream, &dims, &q, &shifted, &values, 7, dims.scale())?;
    assert!(check::max_abs(&base, &shifted_out) < 1e-5);
    Ok(())
}

#[test]
fn gate_matches_oracle_and_extremes() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let d = 32;
    let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 8.0).collect();
    let mut gate: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 4.0).collect();
    gate[..6].copy_from_slice(&[-1000.0, -100.0, -0.0, 0.0, 100.0, 1000.0]);
    let actual = gpu_gate(&stream, &x, &gate, d)?;
    let expected: Vec<f64> = x
        .iter()
        .zip(&gate)
        .map(|(&x, &g)| x as f64 / (1.0 + (-(g as f64)).exp()))
        .collect();
    assert!(check::max_relative(&actual, &expected) < 3e-6);
    Ok(())
}

#[test]
fn cache_append_writes_position_and_heads() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let dims = AttentionDims::new(8, 4, 2, 32)?;
    let mut k =
        api::zeros::<f32>(&[dims.max_seq(), dims.kv_heads(), dims.head_dim()]).sync_on(&stream)?;
    let mut v =
        api::zeros::<f32>(&[dims.max_seq(), dims.kv_heads(), dims.head_dim()]).sync_on(&stream)?;
    for t in 0..3 {
        let key = vec![(t + 1) as f32; dims.kv_heads() * dims.head_dim()];
        let value = vec![-((t + 1) as f32); dims.kv_heads() * dims.head_dim()];
        let key = api::copy_host_vec_to_device(&Arc::new(key))
            .reshape(&[dims.kv_heads(), dims.head_dim()])
            .sync_on(&stream)?;
        let value = api::copy_host_vec_to_device(&Arc::new(value))
            .reshape(&[dims.kv_heads(), dims.head_dim()])
            .sync_on(&stream)?;
        let pos = api::copy_host_vec_to_device(&Arc::new(vec![t as i32])).sync_on(&stream)?;
        cache_append(
            k.partition([dims.max_seq(), 1, dims.head_dim()]),
            v.partition([dims.max_seq(), 1, dims.head_dim()]),
            &key,
            &value,
            &pos,
        )
        .generics(dims.cache_generics())
        .sync_on(&stream)?;
    }
    let kh: Vec<f32> = k.to_host_vec().sync_on(&stream)?;
    let vh: Vec<f32> = v.to_host_vec().sync_on(&stream)?;
    for t in 0..dims.max_seq() {
        for &x in &kh[t * 64..(t + 1) * 64] {
            assert_eq!(x, if t < 3 { (t + 1) as f32 } else { 0.0 });
        }
        for &x in &vh[t * 64..(t + 1) * 64] {
            assert_eq!(x, if t < 3 { -((t + 1) as f32) } else { 0.0 });
        }
    }
    Ok(())
}
