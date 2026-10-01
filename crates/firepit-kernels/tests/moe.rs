use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, ToHostVec};
use cutile::tile_kernel::ToHostVecOp;
use firepit_kernels::dense::DenseMatrix;
use firepit_kernels::moe::{ExpertSelection, MoeDims, combine, expert_gemv};
use firepit_kernels::q8::GemvTile;
use std::sync::Arc;

#[test]
fn top_k_gpu_smoke() -> Result<(), Box<dyn std::error::Error>> {
    let dims = MoeDims::new(32, 32, 32, 5, 3)?;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let logits = api::copy_host_vec_to_device(&Arc::new(vec![0.2f32, 2.0, -1.0, 1.0, 3.0]))
        .sync_on(&stream)?;
    let selection = ExpertSelection::top_k(&logits.view(&[dims.experts()])?, &dims, &stream)?;
    let ids: Vec<i32> = selection.indices().dup().to_host_vec().sync_on(&stream)?;
    let weights: Vec<f32> = selection.weights().dup().to_host_vec().sync_on(&stream)?;
    assert_eq!(&ids[..dims.top_k()], &[4, 1, 3]);
    assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn expert_gemv_gpu_smoke() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let matrix = DenseMatrix::upload(
        (0..2 * 32 * 32).map(|i| (i % 17) as f32 / 17.0).collect(),
        64,
        32,
        &stream,
    )?;
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; 32]))
        .reshape(&[1, 1, 32])
        .sync_on(&stream)?;
    let dims = MoeDims::new(32, 32, 32, 2, 2)?;
    let selection = ExpertSelection::parse(&[1, 0], &[0.5, 0.5], &dims, &stream)?;
    let mut y = api::zeros::<f32>(&[2, 32]).sync_on(&stream)?;
    expert_gemv(
        &matrix,
        &x.view(&[1, 1, 32])?,
        &selection,
        y.partition([1, 16]),
        32,
        0,
        GemvTile {
            rows: 16,
            blocks: 1,
        },
    )
    .sync_on(&stream)?;
    let actual: Vec<f32> = y.to_host_vec().sync_on(&stream)?;
    for rank in 0..2 {
        for row in 0..32 {
            let expected: f32 = (0..32)
                .map(|col| ((ids_host(rank) * 32 * 32 + row * 32 + col) % 17) as f32 / 17.0)
                .sum();
            assert!((actual[rank * 32 + row] - expected).abs() < 1e-4);
        }
    }
    Ok(())
}

fn ids_host(rank: usize) -> usize {
    [1, 0][rank]
}

#[test]
fn expert_selection_parse_rejects_invalid_inputs() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let dims = MoeDims::new(32, 32, 32, 2, 2)?;
    for (ids, weights) in [
        (vec![-1, 0], vec![0.5, 0.5]),
        (vec![0, 2], vec![0.5, 0.5]),
        (vec![0], vec![1.0]),
        (vec![0, 0], vec![0.5, 0.5]),
        (vec![0, 1], vec![f32::NAN, 0.5]),
        (vec![0, 1], vec![f32::INFINITY, 0.5]),
        (vec![0, 1], vec![-0.1, 1.1]),
        (vec![0, 1], vec![0.0, 0.0]),
    ] {
        assert!(ExpertSelection::parse(&ids, &weights, &dims, &stream).is_err());
    }
    assert!(ExpertSelection::parse(&[0, 1], &[0.5], &dims, &stream).is_err());
    Ok(())
}

#[test]
fn combine_gpu_smoke() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let routed = api::copy_host_vec_to_device(&Arc::new(
        vec![1.0f32; 32].into_iter().chain(vec![3.0; 32]).collect(),
    ))
    .reshape(&[2, 32])
    .sync_on(&stream)?;
    let dims = MoeDims::new(32, 32, 32, 2, 2)?;
    let selection = ExpertSelection::parse(&[0, 1], &[0.25, 0.75], &dims, &stream)?;
    let shared = api::copy_host_vec_to_device(&Arc::new(vec![2.0f32; 32]))
        .reshape(&[1, 32])
        .sync_on(&stream)?;
    let gate = api::copy_host_vec_to_device(&Arc::new(vec![0.0f32]))
        .reshape(&[1, 1])
        .sync_on(&stream)?;
    let mut out = api::zeros::<f32>(&[1, 32]).sync_on(&stream)?;
    combine(out.partition([1, 32]), &routed, &selection, &shared, &gate).sync_on(&stream)?;
    let actual: Vec<f32> = out.to_host_vec().sync_on(&stream)?;
    assert!(actual.iter().all(|v| (v - 3.5).abs() < 1e-6));
    Ok(())
}

mod common;
use common::{check, strategy};
use proptest::prelude::*;

fn deep_cases() -> bool {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .is_some_and(|cases| cases >= 256)
}

fn select_gpu(
    stream: &Arc<cuda_core::Stream>,
    logits: &[f32],
    k: usize,
) -> Result<(Vec<i32>, Vec<f32>), Box<dyn std::error::Error>> {
    let dims = MoeDims::new(32, 32, 32, logits.len(), k)?;
    let x = api::copy_host_vec_to_device(&Arc::new(logits.to_vec())).sync_on(stream)?;
    let selection = ExpertSelection::top_k(&x.view(&[logits.len()])?, &dims, stream)?;
    let mut ids: Vec<i32> = selection.indices().dup().to_host_vec().sync_on(stream)?;
    let mut weights: Vec<f32> = selection.weights().dup().to_host_vec().sync_on(stream)?;
    ids.truncate(k);
    weights.truncate(k);
    Ok((ids, weights))
}

#[test]
fn top_k_oracle_and_metamorphic() {
    let stream = common::stream();
    let mut sizes = vec![5usize, 8, 16, 32];
    if deep_cases() {
        sizes.push(256);
    }
    let cases = prop::sample::select(sizes).prop_flat_map(|experts| {
        (
            Just(experts),
            1usize..=experts.min(8),
            strategy::values(experts),
        )
    });
    common::runner(64, file!())
        .run(&cases, |(experts, k, values)| {
            let logits: Vec<f32> = values
                .iter()
                .enumerate()
                .map(|(i, &v)| v + i as f32 * 0.0001)
                .collect();
            let (ids, weights) =
                select_gpu(&stream, &logits, k).map_err(|e| TestCaseError::fail(e.to_string()))?;
            let mut expected: Vec<usize> = (0..experts).collect();
            expected.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            expected.truncate(k);
            let max = logits[expected[0]] as f64;
            let scores: Vec<f64> = expected
                .iter()
                .map(|&id| (logits[id] as f64 - max).exp())
                .collect();
            let total: f64 = scores.iter().sum();
            for rank in 0..k {
                prop_assert_eq!(ids[rank], expected[rank] as i32);
                prop_assert!((weights[rank] as f64 - scores[rank] / total).abs() < 2e-6);
            }
            let unique: std::collections::HashSet<_> = ids.iter().copied().collect();
            prop_assert_eq!(unique.len(), k);
            prop_assert!(ids.iter().all(|&id| id >= 0 && (id as usize) < experts));
            prop_assert!((weights.iter().sum::<f32>() - 1.0).abs() < 2e-6);
            let reversed: Vec<f32> = logits.iter().copied().rev().collect();
            let (permuted, permuted_weights) = select_gpu(&stream, &reversed, k)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            for rank in 0..k {
                prop_assert_eq!(experts - 1 - permuted[rank] as usize, ids[rank] as usize);
                prop_assert!((permuted_weights[rank] - weights[rank]).abs() < 2e-6);
            }
            if k < experts {
                let unused = (0..experts).find(|id| !expected.contains(id)).unwrap();
                let mut changed = logits.clone();
                changed[unused] = -1000.0;
                let (same_ids, same_weights) = select_gpu(&stream, &changed, k)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                prop_assert_eq!(same_ids, ids);
                for rank in 0..k {
                    prop_assert!((same_weights[rank] - weights[rank]).abs() < 2e-6);
                }
            }
            Ok(())
        })
        .unwrap();
}

#[test]
fn expert_gemv_matches_f64_oracle() {
    let stream = common::stream();
    let mut experts = vec![2usize, 4];
    let mut rows = vec![16usize, 32];
    let mut cols = vec![128usize, 256];
    if deep_cases() {
        experts.push(8);
        rows.push(64);
        cols.push(512);
    }
    let cases = (
        prop::sample::select(experts),
        prop::sample::select(rows),
        prop::sample::select(cols),
    )
        .prop_flat_map(|(experts, rows, cols)| {
            (
                Just(experts),
                Just(rows),
                Just(cols),
                1usize..=experts.min(4),
            )
        })
        .prop_flat_map(|(experts, rows, cols, k)| {
            (
                Just((experts, rows, cols, k)),
                strategy::values(experts * rows * cols),
                strategy::values(k * cols),
            )
        });
    common::runner(16, file!())
        .run(&cases, |((experts, rows, cols, k), weights, inputs)| {
            let ids_host: Vec<i32> = (0..k)
                .map(|rank| ((rank * 3 + 1) % experts) as i32)
                .collect();
            let dims = MoeDims::new(cols, 32, 32, experts, k)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let selection = ExpertSelection::parse(&ids_host, &vec![1.0; k], &dims, &stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let matrix = DenseMatrix::upload(weights.clone(), experts * rows, cols, &stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            for stride in [0usize, 1] {
                let input = if stride == 0 {
                    inputs[..cols].to_vec()
                } else {
                    inputs.clone()
                };
                let x = api::copy_host_vec_to_device(&Arc::new(input.clone()))
                    .reshape(&[if stride == 0 { 1 } else { k }, cols / 32, 32])
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let mut out = api::zeros::<f32>(&[k, rows])
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                expert_gemv(
                    &matrix,
                    &x.view(&[if stride == 0 { 1 } else { k }, cols / 32, 32])
                        .map_err(|e| TestCaseError::fail(e.to_string()))?,
                    &selection,
                    out.partition([1, 16]),
                    rows,
                    stride,
                    GemvTile {
                        rows: 16,
                        blocks: 4,
                    },
                )
                .sync_on(&stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let got: Vec<f32> = out
                    .to_host_vec()
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let mut expected = Vec::new();
                let mut magnitude = Vec::new();
                for rank in 0..k {
                    let x = &input[(rank * stride) * cols..(rank * stride + 1) * cols];
                    let expert = ids_host[rank] as usize;
                    for row in 0..rows {
                        let w = &weights
                            [(expert * rows + row) * cols..(expert * rows + row + 1) * cols];
                        expected.push(w.iter().zip(x).map(|(&w, &x)| w as f64 * x as f64).sum());
                        magnitude.push(
                            w.iter()
                                .zip(x)
                                .map(|(&w, &x)| (w as f64 * x as f64).abs())
                                .sum(),
                        );
                    }
                }
                prop_assert!(check::inner_product_excess(&got, &expected, &magnitude, cols) <= 1.0);
            }
            Ok(())
        })
        .unwrap();
}

#[test]
fn combine_matches_f64_oracle() {
    let stream = common::stream();
    let cases = (prop::sample::select(vec![32usize, 128]), 1usize..=8).prop_flat_map(|(h, k)| {
        (
            Just((h, k)),
            strategy::values(h * k),
            strategy::values(k),
            strategy::values(h),
            strategy::values(1),
        )
    });
    common::runner(64, file!())
        .run(&cases, |((h, k), routed, weights, shared, gate)| {
            let routed_gpu = api::copy_host_vec_to_device(&Arc::new(routed.clone()))
                .reshape(&[k, h])
                .sync_on(&stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let scores: Vec<f32> = weights.iter().map(|&weight| weight.abs() + 1.0).collect();
            let dims =
                MoeDims::new(h, 32, 32, k, k).map_err(|e| TestCaseError::fail(e.to_string()))?;
            let ids: Vec<i32> = (0..k as i32).collect();
            let selection = ExpertSelection::parse(&ids, &scores, &dims, &stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let shared_gpu = api::copy_host_vec_to_device(&Arc::new(shared.clone()))
                .reshape(&[1, h])
                .sync_on(&stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let gate_gpu = api::copy_host_vec_to_device(&Arc::new(gate.clone()))
                .reshape(&[1, 1])
                .sync_on(&stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let mut out = api::zeros::<f32>(&[1, h])
                .sync_on(&stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            combine(
                out.partition([1, h]),
                &routed_gpu,
                &selection,
                &shared_gpu,
                &gate_gpu,
            )
            .sync_on(&stream)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let got: Vec<f32> = out
                .to_host_vec()
                .sync_on(&stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let sigmoid = 1.0 / (1.0 + (-(gate[0] as f64)).exp());
            let expected: Vec<f64> = (0..h)
                .map(|col| {
                    (0..k)
                        .map(|rank| scores[rank] as f64 * routed[rank * h + col] as f64)
                        .sum::<f64>()
                        + sigmoid * shared[col] as f64
                })
                .collect();
            let magnitude: Vec<f64> = (0..h)
                .map(|col| {
                    (0..k)
                        .map(|rank| (scores[rank] as f64 * routed[rank * h + col] as f64).abs())
                        .sum::<f64>()
                        + (sigmoid * shared[col] as f64).abs()
                })
                .collect();
            prop_assert!(check::inner_product_excess(&got, &expected, &magnitude, k + 2) <= 1.0);
            Ok(())
        })
        .unwrap();
}
