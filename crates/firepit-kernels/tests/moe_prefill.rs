mod common;

use common::{check, strategy};

use cuda_async::device_operation::DeviceOp;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, ToHostVec};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use firepit_kernels::gemm::{GemmTile, gemm, grouped_gemm};
use firepit_kernels::gemv::{GemvTile, expert_gemv, gemv};
use firepit_kernels::moe::{
    ExpertSelection, MoeDims, PrefillLayout, PrefillPlan, combine, combine_prefill,
    combine_prefill_view, gather_prefill,
};
use firepit_kernels::sink::Eager;
use firepit_kernels::weights::{DenseMatrix, Weight};
use half::bf16;
use proptest::prelude::*;
use std::sync::Arc;

#[test]
fn padded_expert_gemm_smoke() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let dims = MoeDims::new(32, 32, 32, 2, 1)?;
    let tile = GemmTile {
        m: 16,
        n: 32,
        k: 32,
    };
    let plan = PrefillPlan::parse(&[1, 0], &[1.0, 1.0], &dims, tile.m, &stream)?;
    assert_eq!(plan.layout().token_rows()[0], 1);
    assert_eq!(plan.layout().token_rows()[16], 0);
    assert_eq!(plan.layout().tile_experts(), &[0, 1]);
    assert_eq!(plan.layout().inverse(), &[16, 0]);
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; 64]))
        .reshape(&[2, 32])
        .sync_on(&stream)?;
    let gate_up = DenseMatrix::upload(vec![bf16::from_f32(1.0); 2 * 64 * 32], 2 * 64, 32, &stream)?;
    let down = DenseMatrix::upload(vec![bf16::from_f32(1.0); 2 * 32 * 32], 2 * 32, 32, &stream)?;
    let mut sorted = api::zeros::<f32>(&[32, 32]).sync_on(&stream)?;
    gather_prefill(
        sorted.partition([1, tile.k]),
        &x.view(&[2, 32])?,
        &plan,
        tile.k,
    )
    .sync_on(&stream)?;
    let mut projected = api::zeros::<f32>(&[32, 64]).sync_on(&stream)?;
    grouped_gemm(
        &gate_up,
        &sorted.view(&[32, 32])?,
        projected.partition([16, 32]),
        &plan,
        64,
        tile,
    )
    .submit(&mut Eager(&stream))?;
    let projected_host: Vec<f32> = projected.dup().to_host_vec().sync_on(&stream)?;
    assert_eq!(projected_host[0], 32.0);
    assert_eq!(projected_host[16 * 64], 32.0);
    let mut activation = api::zeros::<f32>(&[32, 32]).sync_on(&stream)?;
    firepit_kernels::swiglu::swiglu(activation.partition([1, 32]), &projected)
        .generics(vec!["32".to_string()])
        .sync_on(&stream)?;
    let mut routed = api::zeros::<f32>(&[32, 32]).sync_on(&stream)?;
    grouped_gemm(
        &down,
        &activation.view(&[32, 32])?,
        routed.partition([16, 32]),
        &plan,
        32,
        tile,
    )
    .submit(&mut Eager(&stream))?;
    let shared = api::zeros::<f32>(&[2, 32]).sync_on(&stream)?;
    let gate = api::zeros::<f32>(&[2, 1]).sync_on(&stream)?;
    let mut output = api::zeros::<f32>(&[2, 32]).sync_on(&stream)?;
    combine_prefill(output.partition([1, 32]), &routed, &plan, &shared, &gate).sync_on(&stream)?;
    let actual: Vec<f32> = output.to_host_vec().sync_on(&stream)?;
    assert!(actual.iter().all(|&value| (value - 32768.0).abs() < 1e-3));
    Ok(())
}

#[test]
fn shared_gate_single_row_gemm() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; 16 * 32]))
        .reshape(&[16, 32])
        .sync_on(&stream)?;
    let weight = DenseMatrix::upload(vec![bf16::from_f32(1.0); 32], 1, 32, &stream)?;
    let mut gate = api::zeros::<f32>(&[16, 1]).sync_on(&stream)?;
    gemm(
        &weight,
        &x.view(&[16, 32])?,
        gate.partition([16, 1]),
        GemmTile { m: 16, n: 1, k: 32 },
    )
    .submit(&mut Eager(&stream))?;
    let values: Vec<f32> = gate.to_host_vec().sync_on(&stream)?;
    assert!(values.iter().all(|&v| v == 32.0));
    Ok(())
}

#[test]
fn prefill_matches_tokenwise_decode() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let (tokens, hidden, intermediate, experts, k) = (3, 32, 32, 32, 2);
    let dims = MoeDims::new(hidden, intermediate, intermediate, experts, k)?;
    let tile = GemmTile {
        m: 16,
        n: 32,
        k: 32,
    };
    let x_host: Vec<f32> = (0..tokens * hidden)
        .map(|i| (i % 19 + 1) as f32 / 32.0)
        .collect();
    let router_data: Vec<bf16> = (0..experts * hidden)
        .map(|i| bf16::from_f32((i / hidden + 1) as f32 / 128.0))
        .collect();
    let expert_gate_data: Vec<bf16> = (0..experts * 2 * intermediate * hidden)
        .map(|i| bf16::from_f32((i / (2 * intermediate * hidden) + 1) as f32 / 2048.0))
        .collect();
    let expert_down_data: Vec<bf16> = (0..experts * hidden * intermediate)
        .map(|i| bf16::from_f32((i / (hidden * intermediate) + 1) as f32 / 2048.0))
        .collect();
    let router = DenseMatrix::upload(router_data, experts, hidden, &stream)?;
    let expert_gate = DenseMatrix::upload(
        expert_gate_data,
        experts * 2 * intermediate,
        hidden,
        &stream,
    )?;
    let expert_down =
        DenseMatrix::upload(expert_down_data, experts * hidden, intermediate, &stream)?;
    let shared_gate_up = DenseMatrix::upload(
        vec![bf16::from_f32(0.01); 2 * intermediate * hidden],
        2 * intermediate,
        hidden,
        &stream,
    )?;
    let shared_down = DenseMatrix::upload(
        vec![bf16::from_f32(0.01); hidden * intermediate],
        hidden,
        intermediate,
        &stream,
    )?;
    let shared_gate = DenseMatrix::upload(vec![bf16::from_f32(0.01); hidden], 1, hidden, &stream)?;
    let mut padded_x = x_host.clone();
    padded_x.resize(tile.m * hidden, 0.0);
    let x = api::copy_host_vec_to_device(&Arc::new(padded_x))
        .reshape(&[tile.m, hidden])
        .sync_on(&stream)?;
    let mut logits = api::zeros::<f32>(&[tile.m, experts]).sync_on(&stream)?;
    gemm(
        &router,
        &x.view(&[tile.m, hidden])?,
        logits.partition([tile.m, tile.n]),
        tile,
    )
    .submit(&mut Eager(&stream))?;
    let plan = PrefillPlan::top_k(
        &logits.slice(&[0..tokens, 0..experts])?,
        &dims,
        tile.m,
        &stream,
    )?;
    let exact_x = api::copy_host_vec_to_device(&Arc::new(x_host.clone()))
        .reshape(&[tokens, hidden])
        .sync_on(&stream)?;
    let rows = plan.layout().padded_rows();
    let mut sorted = api::zeros::<f32>(&[rows, hidden]).sync_on(&stream)?;
    gather_prefill(
        sorted.partition([1, tile.k]),
        &exact_x.view(&[tokens, hidden])?,
        &plan,
        tile.k,
    )
    .sync_on(&stream)?;
    let mut projected = api::zeros::<f32>(&[rows, 2 * intermediate]).sync_on(&stream)?;
    grouped_gemm(
        &expert_gate,
        &sorted.view(&[rows, hidden])?,
        projected.partition([tile.m, tile.n]),
        &plan,
        2 * intermediate,
        tile,
    )
    .submit(&mut Eager(&stream))?;
    let mut activated = api::zeros::<f32>(&[rows, intermediate]).sync_on(&stream)?;
    firepit_kernels::swiglu::swiglu(activated.partition([1, intermediate]), &projected)
        .generics(vec![intermediate.to_string()])
        .sync_on(&stream)?;
    let mut routed = api::zeros::<f32>(&[rows, hidden]).sync_on(&stream)?;
    grouped_gemm(
        &expert_down,
        &activated.view(&[rows, intermediate])?,
        routed.partition([tile.m, tile.n]),
        &plan,
        hidden,
        tile,
    )
    .submit(&mut Eager(&stream))?;
    let mut shared_projected = api::zeros::<f32>(&[tile.m, 2 * intermediate]).sync_on(&stream)?;
    gemm(
        &shared_gate_up,
        &x.view(&[tile.m, hidden])?,
        shared_projected.partition([tile.m, tile.n]),
        tile,
    )
    .submit(&mut Eager(&stream))?;
    let mut shared_activated = api::zeros::<f32>(&[tile.m, intermediate]).sync_on(&stream)?;
    firepit_kernels::swiglu::swiglu(
        shared_activated.partition([1, intermediate]),
        &shared_projected,
    )
    .generics(vec![intermediate.to_string()])
    .sync_on(&stream)?;
    let mut shared = api::zeros::<f32>(&[tile.m, hidden]).sync_on(&stream)?;
    gemm(
        &shared_down,
        &shared_activated.view(&[tile.m, intermediate])?,
        shared.partition([tile.m, tile.n]),
        tile,
    )
    .submit(&mut Eager(&stream))?;
    let mut gate = api::zeros::<f32>(&[tile.m, 1]).sync_on(&stream)?;
    gemm(
        &shared_gate,
        &x.view(&[tile.m, hidden])?,
        gate.partition([tile.m, 1]),
        GemmTile { n: 1, ..tile },
    )
    .submit(&mut Eager(&stream))?;
    let mut prefill = api::zeros::<f32>(&[tokens, hidden]).sync_on(&stream)?;
    combine_prefill_view(
        prefill.partition([1, hidden]),
        &routed,
        &plan,
        &shared.slice(&[0..tokens, 0..hidden])?,
        &gate.slice(&[0..tokens, 0..1])?,
    )
    .sync_on(&stream)?;
    let prefill: Vec<f32> = prefill.to_host_vec().sync_on(&stream)?;

    let mut decode = Vec::new();
    let gemv_tile = GemvTile {
        rows: 16,
        blocks: 1,
    };
    for token in 0..tokens {
        let row = api::copy_host_vec_to_device(&Arc::new(
            x_host[token * hidden..(token + 1) * hidden].to_vec(),
        ))
        .reshape(&[1, 1, 32])
        .sync_on(&stream)?;
        let view = row.view(&[1, 1, 32])?;
        let mut router_logits = api::zeros::<f32>(&[1, experts]).sync_on(&stream)?;
        gemv(
            &Weight::Dense(router.clone()),
            &view,
            router_logits.partition([1, gemv_tile.rows]),
            gemv_tile,
        )
        .submit(&mut Eager(&stream))?;
        let selection = ExpertSelection::top_k(&router_logits.view(&[experts])?, &dims, &stream)?;
        let mut projected = api::zeros::<f32>(&[k, 2 * intermediate]).sync_on(&stream)?;
        expert_gemv(
            &expert_gate,
            &view,
            &selection,
            projected.partition([1, gemv_tile.rows]),
            2 * intermediate,
            0,
            gemv_tile,
        )
        .sync_on(&stream)?;
        let mut activated = api::zeros::<f32>(&[k, intermediate]).sync_on(&stream)?;
        firepit_kernels::swiglu::swiglu(activated.partition([1, intermediate]), &projected)
            .generics(vec![intermediate.to_string()])
            .sync_on(&stream)?;
        let mut routed = api::zeros::<f32>(&[k, hidden]).sync_on(&stream)?;
        expert_gemv(
            &expert_down,
            &activated.view(&[k, 1, 32])?,
            &selection,
            routed.partition([1, gemv_tile.rows]),
            hidden,
            1,
            gemv_tile,
        )
        .sync_on(&stream)?;
        let mut shared_projected = api::zeros::<f32>(&[1, 2 * intermediate]).sync_on(&stream)?;
        gemv(
            &Weight::Dense(shared_gate_up.clone()),
            &view,
            shared_projected.partition([1, gemv_tile.rows]),
            gemv_tile,
        )
        .submit(&mut Eager(&stream))?;
        let mut shared_activated = api::zeros::<f32>(&[1, intermediate]).sync_on(&stream)?;
        firepit_kernels::swiglu::swiglu(
            shared_activated.partition([1, intermediate]),
            &shared_projected,
        )
        .generics(vec![intermediate.to_string()])
        .sync_on(&stream)?;
        let mut shared_output = api::zeros::<f32>(&[1, hidden]).sync_on(&stream)?;
        gemv(
            &Weight::Dense(shared_down.clone()),
            &shared_activated.view(&[1, 1, 32])?,
            shared_output.partition([1, gemv_tile.rows]),
            gemv_tile,
        )
        .submit(&mut Eager(&stream))?;
        let mut gate_output = api::zeros::<f32>(&[1, 1]).sync_on(&stream)?;
        gemv(
            &Weight::Dense(shared_gate.clone()),
            &view,
            gate_output.partition([1, 1]),
            GemvTile { rows: 1, blocks: 1 },
        )
        .submit(&mut Eager(&stream))?;
        let mut output = api::zeros::<f32>(&[1, hidden]).sync_on(&stream)?;
        combine(
            output.partition([1, hidden]),
            &routed,
            &selection,
            &shared_output,
            &gate_output,
        )
        .sync_on(&stream)?;
        decode.extend(output.to_host_vec().sync_on(&stream)?);
    }
    let bound =
        4.0 * (check::gamma_tilde(hidden) + check::gamma_tilde(intermediate)) + 2.0 / 65536.0;
    for (&actual, &expected) in prefill.iter().zip(&decode) {
        assert!(
            (actual as f64 - expected as f64).abs()
                <= bound * (actual as f64).abs().max((expected as f64).abs())
        );
    }
    Ok(())
}

#[test]
fn layout_rejects_invalid() -> Result<(), Box<dyn std::error::Error>> {
    let dims = MoeDims::new(32, 32, 32, 3, 2)?;
    for (ids, weights) in [
        (vec![], vec![]),
        (vec![0], vec![1.0]),
        (vec![0, 3], vec![0.5, 0.5]),
        (vec![0, 0], vec![0.5, 0.5]),
        (vec![0, 1], vec![0.0, 0.0]),
        (vec![0, 1], vec![f32::NAN, 1.0]),
    ] {
        assert!(PrefillLayout::parse(&ids, &weights, &dims, 16).is_err());
    }
    Ok(())
}

#[test]
fn layout_covers_single_token_skew_and_empty_experts() -> Result<(), Box<dyn std::error::Error>> {
    let single = MoeDims::new(32, 32, 32, 8, 1)?;
    let layout = PrefillLayout::parse(&[7], &[1.0], &single, 16)?;
    assert_eq!(layout.padded_rows(), 16);
    assert_eq!(layout.padding_rows(), 15);
    assert_eq!(layout.tile_experts(), &[7]);
    assert!(layout.ranges()[..7].iter().all(|&(_, count)| count == 0));
    let skew = MoeDims::new(32, 32, 32, 8, 2)?;
    let layout = PrefillLayout::parse(&[0, 1, 0, 1, 0, 1], &[0.5; 6], &skew, 16)?;
    assert_eq!(layout.ranges()[0].1, 3);
    assert_eq!(layout.ranges()[1].1, 3);
    assert!(layout.ranges()[2..].iter().all(|&(_, count)| count == 0));
    Ok(())
}

#[test]
fn layout_oracle_pbt() {
    let cases = (1usize..=33, 1usize..=3, any::<bool>(), any::<u16>());
    common::runner(64, file!())
        .run(&cases, |(tokens, k, skew, seed)| {
            let dims = MoeDims::new(32, 32, 32, 4, k).unwrap();
            let ids: Vec<i32> = (0..tokens)
                .flat_map(|token| {
                    (0..k).map(move |slot| {
                        ((if skew { 0 } else { token + seed as usize }) + slot) as i32 % 4
                    })
                })
                .collect();
            let scores = vec![1.0 / k as f32; tokens * k];
            let layout = PrefillLayout::parse(&ids, &scores, &dims, 16).unwrap();
            let mut expected = Vec::new();
            for expert in 0..4 {
                let mut count = 0usize;
                let start = expected.len();
                for (position, &id) in ids.iter().enumerate() {
                    if id as usize == expert {
                        expected.push((position / k) as i32);
                        count += 1;
                    }
                }
                expected.resize(start + count.div_ceil(16) * 16, -1);
                prop_assert_eq!(layout.ranges()[expert], (start, count));
            }
            prop_assert_eq!(layout.token_rows(), expected);
            prop_assert_eq!(layout.padded_rows() % 16, 0);
            prop_assert_eq!(layout.padding_rows(), layout.padded_rows() - tokens * k);
            for (tile, &expert) in layout.tile_experts().iter().enumerate() {
                for row in tile * 16..(tile + 1) * 16 {
                    let token = layout.token_rows()[row];
                    if token >= 0 {
                        prop_assert!(
                            ids[token as usize * k..(token as usize + 1) * k].contains(&expert)
                        );
                    }
                }
            }
            for (position, &row) in layout.inverse().iter().enumerate() {
                prop_assert_eq!(layout.token_rows()[row as usize], (position / k) as i32);
                prop_assert_eq!(layout.tile_experts()[row as usize / 16], ids[position]);
            }
            Ok(())
        })
        .unwrap();
}

#[test]
fn gather_prefill_oracle_pbt() {
    let stream = common::stream();
    let cases = (
        1usize..=17,
        1usize..=4,
        any::<bool>(),
        prop::sample::select(vec![16usize, 32]),
    )
        .prop_flat_map(|(tokens, k, skew, tile_rows)| {
            (
                Just((tokens, k, skew, tile_rows)),
                strategy::values(tokens * 32),
            )
        });
    common::runner(64, file!())
        .run(&cases, |((tokens, k, skew, tile_rows), input)| {
            let fail = |e: Box<dyn std::error::Error>| TestCaseError::fail(e.to_string());
            let dims =
                MoeDims::new(32, 32, 32, 4, k).map_err(|e| TestCaseError::fail(e.to_string()))?;
            let ids: Vec<i32> = (0..tokens)
                .flat_map(|token| {
                    (0..k).map(move |slot| ((if skew { 0 } else { token }) + slot) as i32 % 4)
                })
                .collect();
            let plan = PrefillPlan::parse(&ids, &vec![1.0; ids.len()], &dims, tile_rows, &stream)
                .map_err(fail)?;
            let x = api::copy_host_vec_to_device(&Arc::new(input.clone()))
                .reshape(&[tokens, 32])
                .sync_on(&stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let mut sorted = api::zeros::<f32>(&[plan.layout().padded_rows(), 32])
                .sync_on(&stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            gather_prefill(
                sorted.partition([1, 32]),
                &x.view(&[tokens, 32])
                    .map_err(|e| TestCaseError::fail(e.to_string()))?,
                &plan,
                32,
            )
            .sync_on(&stream)
            .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let actual: Vec<f32> = sorted
                .to_host_vec()
                .sync_on(&stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            for (row, &token) in plan.layout().token_rows().iter().enumerate() {
                let expected = if token < 0 {
                    vec![0.0; 32]
                } else {
                    input[token as usize * 32..(token as usize + 1) * 32].to_vec()
                };
                prop_assert_eq!(&actual[row * 32..(row + 1) * 32], &expected);
            }
            Ok(())
        })
        .unwrap();
}

#[test]
fn top_k_prefill_matches_decode() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let dims = MoeDims::new(32, 32, 32, 5, 3)?;
    let scores: Vec<f32> = (0..7)
        .flat_map(|token| (0..5).map(move |expert| ((token * 7 + expert * 13) % 19) as f32 / 7.0))
        .collect();
    let logits = api::copy_host_vec_to_device(&Arc::new(scores.clone()))
        .reshape(&[7, 5])
        .sync_on(&stream)?;
    let plan = PrefillPlan::top_k(&logits.view(&[7, 5])?, &dims, 16, &stream)?;
    let ids: Vec<i32> = plan.indices().dup().to_host_vec().sync_on(&stream)?;
    let weights: Vec<f32> = plan.weights().dup().to_host_vec().sync_on(&stream)?;
    for token in 0..7 {
        let row =
            api::copy_host_vec_to_device(&Arc::new(scores[token * 5..(token + 1) * 5].to_vec()))
                .sync_on(&stream)?;
        let decode = ExpertSelection::top_k(&row.view(&[5])?, &dims, &stream)?;
        let expected_ids: Vec<i32> = decode.indices().dup().to_host_vec().sync_on(&stream)?;
        let expected_weights: Vec<f32> = decode.weights().dup().to_host_vec().sync_on(&stream)?;
        assert_eq!(&ids[token * 4..token * 4 + 3], &expected_ids[..3]);
        for slot in 0..3 {
            assert!((weights[token * 4 + slot] - expected_weights[slot]).abs() < 1e-6);
        }
    }
    Ok(())
}

#[test]
fn grouped_gemm_and_combine_oracle_pbt() {
    let stream = common::stream();
    let cases = (1usize..=17, 1usize..=3, any::<bool>()).prop_flat_map(|(tokens, k, skew)| {
        (
            Just((tokens, k, skew)),
            strategy::values(tokens * 32),
            strategy::values(4 * 64 * 32),
            strategy::values(4 * 32 * 32),
            strategy::values(tokens * 32),
        )
    });
    common::runner(64, file!())
        .run(
            &cases,
            |((tokens, k, skew), x, gate_values, down_values, shared)| {
                let fail = |e: Box<dyn std::error::Error>| TestCaseError::fail(e.to_string());
                let dims = MoeDims::new(32, 32, 32, 4, k)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let tile = GemmTile {
                    m: 16,
                    n: 32,
                    k: 32,
                };
                let ids: Vec<i32> = (0..tokens)
                    .flat_map(|token| {
                        (0..k).map(move |slot| ((if skew { 0 } else { token }) + slot) as i32 % 4)
                    })
                    .collect();
                let normalizer = (k * (k + 1) / 2) as f32;
                let scores: Vec<f32> = (0..tokens)
                    .flat_map(|_| (1..=k).map(|slot| slot as f32 / normalizer))
                    .collect();
                let plan = PrefillPlan::parse(&ids, &scores, &dims, 16, &stream).map_err(fail)?;
                let x: Vec<f32> = x.into_iter().map(|v| v / 16.0).collect();
                let shared: Vec<f32> = shared.into_iter().map(|v| v / 16.0).collect();
                let gate_values: Vec<bf16> = gate_values
                    .into_iter()
                    .map(|v| bf16::from_f32(v / 16.0))
                    .collect();
                let down_values: Vec<bf16> = down_values
                    .into_iter()
                    .map(|v| bf16::from_f32(v / 16.0))
                    .collect();
                let gate_f32: Vec<f32> = gate_values.iter().map(|v| v.to_f32()).collect();
                let down_f32: Vec<f32> = down_values.iter().map(|v| v.to_f32()).collect();
                let x_gpu = api::copy_host_vec_to_device(&Arc::new(x.clone()))
                    .reshape(&[tokens, 32])
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let gate_weight = DenseMatrix::upload(gate_values, 4 * 64, 32, &stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let down_weight = DenseMatrix::upload(down_values, 4 * 32, 32, &stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let padded = plan.layout().padded_rows();
                let mut sorted = api::zeros::<f32>(&[padded, 32])
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                gather_prefill(
                    sorted.partition([1, tile.k]),
                    &x_gpu
                        .view(&[tokens, 32])
                        .map_err(|e| TestCaseError::fail(e.to_string()))?,
                    &plan,
                    tile.k,
                )
                .sync_on(&stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let mut projected = api::zeros::<f32>(&[padded, 64])
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                grouped_gemm(
                    &gate_weight,
                    &sorted
                        .view(&[padded, 32])
                        .map_err(|e| TestCaseError::fail(e.to_string()))?,
                    projected.partition([16, 32]),
                    &plan,
                    64,
                    tile,
                )
                .submit(&mut Eager(&stream))
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let projected_host: Vec<f32> = projected
                    .dup()
                    .to_host_vec()
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let split_error = 1.0 / 65536.0;
                for (slot, &row) in plan.layout().inverse().iter().enumerate() {
                    let token = slot / k;
                    let expert = ids[slot] as usize;
                    for output in 0..64 {
                        let weights =
                            &gate_f32[(expert * 64 + output) * 32..(expert * 64 + output + 1) * 32];
                        let values = &x[token * 32..(token + 1) * 32];
                        let expected: f64 = weights
                            .iter()
                            .zip(values)
                            .map(|(&w, &v)| w as f64 * v as f64)
                            .sum();
                        let magnitude: f64 = weights
                            .iter()
                            .zip(values)
                            .map(|(&w, &v)| (w as f64 * v as f64).abs())
                            .sum();
                        let bound = (check::gamma_tilde(32) + split_error) * magnitude;
                        prop_assert!(
                            (projected_host[row as usize * 64 + output] as f64 - expected).abs()
                                <= bound.max(f64::MIN_POSITIVE)
                        );
                    }
                }
                let mut activation = api::zeros::<f32>(&[padded, 32])
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                firepit_kernels::swiglu::swiglu(activation.partition([1, 32]), &projected)
                    .generics(vec!["32".to_string()])
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let activation_host: Vec<f32> = activation
                    .dup()
                    .to_host_vec()
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let mut routed = api::zeros::<f32>(&[padded, 32])
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                grouped_gemm(
                    &down_weight,
                    &activation
                        .view(&[padded, 32])
                        .map_err(|e| TestCaseError::fail(e.to_string()))?,
                    routed.partition([16, 32]),
                    &plan,
                    32,
                    tile,
                )
                .submit(&mut Eager(&stream))
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let routed_host: Vec<f32> = routed
                    .dup()
                    .to_host_vec()
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                for (slot, &row) in plan.layout().inverse().iter().enumerate() {
                    let expert = ids[slot] as usize;
                    for output in 0..32 {
                        let weights =
                            &down_f32[(expert * 32 + output) * 32..(expert * 32 + output + 1) * 32];
                        let values = &activation_host[row as usize * 32..(row as usize + 1) * 32];
                        let expected: f64 = weights
                            .iter()
                            .zip(values)
                            .map(|(&w, &v)| w as f64 * v as f64)
                            .sum();
                        let magnitude: f64 = weights
                            .iter()
                            .zip(values)
                            .map(|(&w, &v)| (w as f64 * v as f64).abs())
                            .sum();
                        let bound = (check::gamma_tilde(32) + split_error) * magnitude;
                        prop_assert!(
                            (routed_host[row as usize * 32 + output] as f64 - expected).abs()
                                <= bound.max(f64::MIN_POSITIVE)
                        );
                    }
                }
                let shared_gpu = api::copy_host_vec_to_device(&Arc::new(shared.clone()))
                    .reshape(&[tokens, 32])
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let gate = api::zeros::<f32>(&[tokens, 1])
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let mut output = api::zeros::<f32>(&[tokens, 32])
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                combine_prefill(
                    output.partition([1, 32]),
                    &routed,
                    &plan,
                    &shared_gpu,
                    &gate,
                )
                .sync_on(&stream)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let actual: Vec<f32> = output
                    .to_host_vec()
                    .sync_on(&stream)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                for token in 0..tokens {
                    for output in 0..32 {
                        let mut expected = shared[token * 32 + output] as f64 * 0.5;
                        let mut magnitude = expected.abs();
                        for slot in 0..k {
                            let row = plan.layout().inverse()[token * k + slot] as usize;
                            let term = scores[token * k + slot] as f64
                                * routed_host[row * 32 + output] as f64;
                            expected += term;
                            magnitude += term.abs();
                        }
                        let bound = check::gamma_tilde(k + 2) * magnitude;
                        prop_assert!(
                            (actual[token * 32 + output] as f64 - expected).abs()
                                <= bound.max(f64::MIN_POSITIVE)
                        );
                    }
                }
                Ok(())
            },
        )
        .unwrap();
}
