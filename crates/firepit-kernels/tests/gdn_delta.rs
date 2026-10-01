mod common;

use common::{check, strategy};
use cuda_async::device_operation::DeviceOp;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, ToHostVec};
use cutile::tile_kernel::TileKernel;
use firepit_kernels::gdn::delta_step;
use proptest::prelude::*;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
struct Shape {
    kh: usize,
    vh: usize,
    dk: usize,
    dv: usize,
    tile: usize,
}

fn gpu_step(
    stream: &Arc<cuda_core::Stream>,
    shape: Shape,
    state: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: &[f32],
    a: &[f32],
    a_log: &[f32],
    dt: &[f32],
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    let upload =
        |data: &[f32]| api::copy_host_vec_to_device(&Arc::new(data.to_vec())).sync_on(stream);
    let q = api::copy_host_vec_to_device(&Arc::new(q.to_vec()))
        .reshape(&[shape.kh, shape.dk])
        .sync_on(stream)?;
    let k = api::copy_host_vec_to_device(&Arc::new(k.to_vec()))
        .reshape(&[shape.kh, shape.dk])
        .sync_on(stream)?;
    let v = api::copy_host_vec_to_device(&Arc::new(v.to_vec()))
        .reshape(&[shape.vh, shape.dv])
        .sync_on(stream)?;
    let b = upload(b)?;
    let a = upload(a)?;
    let a_log = upload(a_log)?;
    let dt = upload(dt)?;
    let mut state = api::copy_host_vec_to_device(&Arc::new(state.to_vec()))
        .reshape(&[shape.vh, shape.dk, shape.dv])
        .sync_on(stream)?;
    let mut out = api::zeros::<f32>(&[shape.vh, 1, shape.dv]).sync_on(stream)?;
    delta_step(
        (&mut out).partition([1, 1, shape.tile]),
        state.partition([1, shape.dk, shape.tile]),
        &q,
        &k,
        &v,
        &b,
        &a,
        &a_log,
        &dt,
        eps,
        1.0 / (shape.dk as f32).sqrt(),
    )
    .generics(vec![
        shape.kh.to_string(),
        shape.vh.to_string(),
        shape.dk.to_string(),
        shape.tile.to_string(),
    ])
    .sync_on(stream)?;
    Ok((
        out.to_host_vec().sync_on(stream)?,
        state.to_host_vec().sync_on(stream)?,
    ))
}

fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

fn oracle_step(
    shape: Shape,
    state: &mut [f64],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: &[f32],
    a: &[f32],
    a_log: &[f32],
    dt: &[f32],
    eps: f32,
    normalize: bool,
    modulo: bool,
) -> Vec<f64> {
    let mut out = vec![0.0; shape.vh * shape.dv];
    for h in 0..shape.vh {
        let key_head = if modulo {
            h % shape.kh
        } else {
            h / (shape.vh / shape.kh)
        };
        let qr = &q[key_head * shape.dk..(key_head + 1) * shape.dk];
        let kr = &k[key_head * shape.dk..(key_head + 1) * shape.dk];
        let qnorm = (qr.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() + eps as f64).sqrt();
        let knorm = (kr.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() + eps as f64).sqrt();
        let beta = 1.0 / (1.0 + (-(b[h] as f64)).exp());
        let decay = (-(a_log[h] as f64).exp() * softplus(a[h] as f64 + dt[h] as f64)).exp();
        for j in 0..shape.dv {
            let mut memory = 0.0;
            for i in 0..shape.dk {
                let p = (h * shape.dk + i) * shape.dv + j;
                state[p] *= decay;
                memory += state[p] * k[key_head * shape.dk + i] as f64
                    / if normalize { knorm } else { 1.0 };
            }
            let delta = beta * (v[h * shape.dv + j] as f64 - memory);
            let mut result = 0.0;
            for i in 0..shape.dk {
                let p = (h * shape.dk + i) * shape.dv + j;
                state[p] +=
                    k[key_head * shape.dk + i] as f64 / if normalize { knorm } else { 1.0 } * delta;
                result += state[p] * q[key_head * shape.dk + i] as f64
                    / if normalize { qnorm } else { 1.0 }
                    / (shape.dk as f64).sqrt();
            }
            out[h * shape.dv + j] = result;
        }
    }
    out
}

#[test]
fn delta_multistep_matches_f64_oracle() {
    let stream = common::stream();
    common::runner(64, file!())
        .run(
            &prop::sample::select(vec![
                Shape {
                    kh: 1,
                    vh: 1,
                    dk: 32,
                    dv: 32,
                    tile: 16,
                },
                Shape {
                    kh: 1,
                    vh: 2,
                    dk: 32,
                    dv: 64,
                    tile: 32,
                },
                Shape {
                    kh: 2,
                    vh: 4,
                    dk: 64,
                    dv: 32,
                    tile: 16,
                },
                Shape {
                    kh: 2,
                    vh: 4,
                    dk: 32,
                    dv: 64,
                    tile: 32,
                },
            ])
            .prop_flat_map(|shape| {
                (
                    Just(shape),
                    strategy::values(shape.vh * shape.dk * shape.dv),
                    strategy::values(3 * shape.kh * shape.dk),
                    strategy::values(3 * shape.kh * shape.dk),
                    strategy::values(3 * shape.vh * shape.dv),
                    strategy::values(3 * shape.vh),
                    strategy::values(3 * shape.vh),
                    strategy::values(shape.vh),
                    strategy::values(shape.vh),
                )
            }),
            |(shape, initial, queries, keys, values, bs, as_, a_log, dt)| {
                let mut gpu_state = initial.clone();
                let mut cpu_state: Vec<f64> = initial.iter().map(|&x| x as f64).collect();
                for step in 0..3 {
                    let q = &queries[step * shape.kh * shape.dk..(step + 1) * shape.kh * shape.dk];
                    let k = &keys[step * shape.kh * shape.dk..(step + 1) * shape.kh * shape.dk];
                    let v = &values[step * shape.vh * shape.dv..(step + 1) * shape.vh * shape.dv];
                    let b = &bs[step * shape.vh..(step + 1) * shape.vh];
                    let a = &as_[step * shape.vh..(step + 1) * shape.vh];
                    let (actual, next) =
                        gpu_step(&stream, shape, &gpu_state, q, k, v, b, a, &a_log, &dt, 1e-6)
                            .map_err(|e| TestCaseError::fail(e.to_string()))?;
                    let expected = oracle_step(
                        shape,
                        &mut cpu_state,
                        q,
                        k,
                        v,
                        b,
                        a,
                        &a_log,
                        &dt,
                        1e-6,
                        true,
                        false,
                    );
                    let output_error = check::max_relative(&actual, &expected);
                    let state_error = check::max_relative(&next, &cpu_state);
                    prop_assert!(
                        output_error < 3e-4,
                        "step {step} output_error={output_error}"
                    );
                    prop_assert!(state_error < 3e-4, "step {step} state_error={state_error}");
                    gpu_state = next;
                }
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn decay_zero_beta_orthogonal_keys_and_tile_invariance() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let shape = Shape {
        kh: 1,
        vh: 2,
        dk: 32,
        dv: 64,
        tile: 16,
    };
    let initial: Vec<f32> = (0..shape.vh * shape.dk * shape.dv)
        .map(|i| (i % 13) as f32 / 17.0)
        .collect();
    let q = vec![1.0; shape.kh * shape.dk];
    let k = vec![0.5; shape.kh * shape.dk];
    let v = vec![1.0; shape.vh * shape.dv];
    let b = vec![-100.0; shape.vh];
    let a = vec![1.0; shape.vh];
    let a_log = vec![0.0; shape.vh];
    let dt = vec![0.0; shape.vh];
    let (out, state) = gpu_step(
        &stream, shape, &initial, &q, &k, &v, &b, &a, &a_log, &dt, 1e-6,
    )?;
    let decay = (-softplus(1.0)).exp() as f32;
    let decayed: Vec<f32> = initial.iter().map(|&x| x * decay).collect();
    assert!(check::max_abs(&state, &decayed) < 2e-6);
    let mut oracle_state: Vec<f64> = initial.iter().map(|&x| x as f64).collect();
    let expected = oracle_step(
        shape,
        &mut oracle_state,
        &q,
        &k,
        &v,
        &b,
        &a,
        &a_log,
        &dt,
        1e-6,
        true,
        false,
    );
    assert!(check::max_relative(&out, &expected) < 2e-5);
    let wider = Shape { tile: 32, ..shape };
    let (out_wide, state_wide) = gpu_step(
        &stream, wider, &initial, &q, &k, &v, &b, &a, &a_log, &dt, 1e-6,
    )?;
    assert!(check::max_abs(&out, &out_wide) < 1e-6);
    assert_eq!(state, state_wide);

    let one = Shape {
        kh: 1,
        vh: 1,
        dk: 32,
        dv: 32,
        tile: 32,
    };
    let zero = vec![0.0; one.dk * one.dv];
    let mut q0 = vec![0.0; one.dk];
    q0[0] = 1.0;
    let mut q1 = vec![0.0; one.dk];
    q1[1] = 1.0;
    let first_v = vec![2.0; one.dv];
    let second_v = vec![3.0; one.dv];
    let (first, state1) = gpu_step(
        &stream,
        one,
        &zero,
        &q0,
        &q0,
        &first_v,
        &[100.0],
        &[-100.0],
        &[0.0],
        &[0.0],
        1e-6,
    )?;
    let (second, _) = gpu_step(
        &stream,
        one,
        &state1,
        &q1,
        &q1,
        &second_v,
        &[100.0],
        &[-100.0],
        &[0.0],
        &[0.0],
        1e-6,
    )?;
    let scale = 1.0 / (one.dk as f32).sqrt();
    assert!((first[0] - 2.0 * scale).abs() < 1e-5);
    assert!((second[0] - 3.0 * scale).abs() < 1e-5);
    Ok(())
}

#[test]
fn gate_extremes_and_wrong_math_are_detected() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let shape = Shape {
        kh: 2,
        vh: 4,
        dk: 32,
        dv: 32,
        tile: 16,
    };
    let state: Vec<f32> = (0..shape.vh * shape.dk * shape.dv)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) / 16.0)
        .collect();
    let q: Vec<f32> = (0..shape.kh * shape.dk)
        .map(|i| ((i * 3 % 19) as f32 - 9.0) / 7.0)
        .collect();
    let k: Vec<f32> = (0..shape.kh * shape.dk)
        .map(|i| ((i * 5 % 17) as f32 - 8.0) / 9.0)
        .collect();
    let v = vec![1.0; shape.vh * shape.dv];
    let b = vec![-100.0, 100.0, -100.0, 100.0];
    let a = vec![100.0, -100.0, 50.0, -50.0];
    let a_log = vec![0.0; shape.vh];
    let dt = vec![0.0; shape.vh];
    let (actual, updated) = gpu_step(
        &stream, shape, &state, &q, &k, &v, &b, &a, &a_log, &dt, 1e-6,
    )?;
    assert!(actual.iter().all(|x| x.is_finite()));
    assert!(updated.iter().all(|x| x.is_finite()));
    let initial: Vec<f64> = state.iter().map(|&x| x as f64).collect();
    let mut right = initial.clone();
    let expected = oracle_step(
        shape, &mut right, &q, &k, &v, &b, &a, &a_log, &dt, 1e-6, true, false,
    );
    assert!(check::max_relative(&actual, &expected) < 2e-5);
    let mut no_norm = initial.clone();
    let wrong_norm = oracle_step(
        shape,
        &mut no_norm,
        &q,
        &k,
        &v,
        &b,
        &a,
        &a_log,
        &dt,
        1e-6,
        false,
        false,
    );
    let mut wrong_map_state = initial;
    let wrong_map = oracle_step(
        shape,
        &mut wrong_map_state,
        &q,
        &k,
        &v,
        &b,
        &a,
        &a_log,
        &dt,
        1e-6,
        true,
        true,
    );
    let no_norm_error = check::max_relative(&actual, &wrong_norm);
    let wrong_map_error = check::max_relative(&actual, &wrong_map);
    assert!(no_norm_error > 1e-3 && wrong_map_error > 1e-3);
    println!("no_l2norm_relative={no_norm_error:.8e} modulo_head_relative={wrong_map_error:.8e}");

    let sensitive = Shape {
        kh: 1,
        vh: 1,
        dk: 32,
        dv: 32,
        tile: 16,
    };
    let (_, sensitive_state) = gpu_step(
        &stream,
        sensitive,
        &vec![1.0; sensitive.dk * sensitive.dv],
        &vec![1.0; sensitive.dk],
        &vec![1.0; sensitive.dk],
        &vec![0.0; sensitive.dv],
        &[-100.0],
        &[-32.0],
        &[32.0],
        &[0.0],
        1e-6,
    )?;
    let expected_decay = (-(32.0f64).exp() * softplus(-32.0)).exp();
    assert!((sensitive_state[0] as f64 - expected_decay).abs() < 2e-6);
    Ok(())
}
