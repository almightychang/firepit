mod common;

use cuda_async::device_operation::DeviceOp;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, ToHostVec};
use cutile::tile_kernel::TileKernel;
use firepit_kernels::gdn::delta_step;
use firepit_kernels::gdn_chunk::{
    apply_inverse, first_phase_matrices, invert_diagonal, merge_inverse, second_phase,
};
use proptest::prelude::*;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
struct Shape {
    chunks: usize,
    chunk: usize,
    tile: usize,
    block: usize,
    kh: usize,
    vh: usize,
    dk: usize,
    dv: usize,
}

struct Inputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
}

struct Outputs {
    u: Vec<f32>,
    w: Vec<f32>,
    intra: Vec<f32>,
    cum: Vec<f32>,
}

struct Reference {
    u: Vec<f64>,
    w: Vec<f64>,
    intra: Vec<f64>,
    cum: Vec<f64>,
    u_rhs_scale: Vec<f64>,
    w_rhs_scale: Vec<f64>,
    u_solve_scale: Vec<f64>,
    w_solve_scale: Vec<f64>,
    intra_scale: Vec<f64>,
    intra_decay_scale: Vec<f64>,
}

fn oracle(shape: Shape, input: &Inputs) -> Reference {
    let (n, c, kh, vh, dk, dv) = (
        shape.chunks,
        shape.tile,
        shape.kh,
        shape.vh,
        shape.dk,
        shape.dv,
    );
    let mut reference = Reference {
        u: vec![0.0; n * vh * c * dv],
        w: vec![0.0; n * vh * c * dk],
        intra: vec![0.0; n * vh * c * c],
        cum: vec![0.0; n * vh * c],
        u_rhs_scale: vec![0.0; n * vh * c * dv],
        w_rhs_scale: vec![0.0; n * vh * c * dk],
        u_solve_scale: vec![0.0; n * vh * c * dv],
        w_solve_scale: vec![0.0; n * vh * c * dk],
        intra_scale: vec![0.0; n * vh * c * c],
        intra_decay_scale: vec![0.0; n * vh * c * c],
    };
    for chunk in 0..n {
        for head in 0..vh {
            let flat = chunk * vh + head;
            let key_head = head / (vh / kh);
            let mut q = vec![0.0f64; c * dk];
            let mut k = vec![0.0f64; c * dk];
            let mut v = vec![0.0f64; c * dv];
            let mut beta = vec![0.0f64; c];
            let mut cum = vec![0.0f64; c];
            let mut gate_magnitude = vec![0.0f64; c];
            for row in 0..c {
                let time = chunk * c + row;
                if row < shape.chunk {
                    beta[row] = input.beta[time * vh + head] as f64;
                    cum[row] = input.g[time * vh + head] as f64;
                    for d in 0..dv {
                        v[row * dv + d] = input.v[(time * vh + head) * dv + d] as f64;
                    }
                    for d in 0..dk {
                        q[row * dk + d] = input.q[(time * kh + key_head) * dk + d] as f64;
                        k[row * dk + d] = input.k[(time * kh + key_head) * dk + d] as f64;
                    }
                }
                if row > 0 {
                    cum[row] += cum[row - 1];
                    gate_magnitude[row] = gate_magnitude[row - 1];
                }
                if row < shape.chunk {
                    gate_magnitude[row] += (input.g[time * vh + head] as f64).abs();
                }
                reference.cum[flat * c + row] = cum[row];
                let qnorm = (q[row * dk..(row + 1) * dk]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f64>()
                    + 1e-6)
                    .sqrt();
                let knorm = (k[row * dk..(row + 1) * dk]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f64>()
                    + 1e-6)
                    .sqrt();
                for d in 0..dk {
                    q[row * dk + d] /= qnorm * (dk as f64).sqrt();
                    k[row * dk + d] /= knorm;
                }
            }
            let mut lower = vec![0.0f64; c * c];
            let mut lower_magnitude = vec![0.0f64; c * c];
            for i in 0..c {
                for j in 0..=i {
                    let decay = (cum[i] - cum[j]).exp();
                    let kk = (0..dk).map(|d| k[i * dk + d] * k[j * dk + d]).sum::<f64>();
                    let qk = (0..dk).map(|d| q[i * dk + d] * k[j * dk + d]).sum::<f64>();
                    let qk_abs = (0..dk)
                        .map(|d| (q[i * dk + d] * k[j * dk + d]).abs())
                        .sum::<f64>();
                    reference.intra[flat * c * c + i * c + j] = qk * decay;
                    reference.intra_scale[flat * c * c + i * c + j] = qk_abs * decay;
                    reference.intra_decay_scale[flat * c * c + i * c + j] =
                        qk_abs * decay * (gate_magnitude[i] + gate_magnitude[j]);
                    if i > j {
                        lower[i * c + j] = beta[i] * kk * decay;
                        lower_magnitude[i * c + j] = beta[i].abs()
                            * decay
                            * (0..dk)
                                .map(|d| (k[i * dk + d] * k[j * dk + d]).abs())
                                .sum::<f64>();
                    }
                }
            }
            for i in 0..c {
                for d in 0..dv {
                    let prior = (0..i)
                        .map(|j| lower[i * c + j] * reference.u[flat * c * dv + j * dv + d])
                        .sum::<f64>();
                    reference.u[flat * c * dv + i * dv + d] = beta[i] * v[i * dv + d] - prior;
                }
                for d in 0..dk {
                    let prior = (0..i)
                        .map(|j| lower[i * c + j] * reference.w[flat * c * dk + j * dk + d])
                        .sum::<f64>();
                    reference.w[flat * c * dk + i * dk + d] =
                        beta[i] * k[i * dk + d] * cum[i].exp() - prior;
                }
            }
            let mut inverse = vec![0.0f64; c * c];
            for i in 0..c {
                inverse[i * c + i] = 1.0;
                for j in 0..i {
                    inverse[i * c + j] = -(j..i)
                        .map(|p| lower[i * c + p] * inverse[p * c + j])
                        .sum::<f64>();
                }
            }
            for i in 0..c {
                for j in 0..=i {
                    let gain = inverse[i * c + j].abs();
                    for d in 0..dv {
                        let rhs = beta[j] * v[j * dv + d];
                        reference.u_rhs_scale[flat * c * dv + i * dv + d] += gain * rhs.abs();
                        let lower_term = (0..j)
                            .map(|p| {
                                (lower[j * c + p].abs() + lower_magnitude[j * c + p])
                                    * reference.u[flat * c * dv + p * dv + d].abs()
                            })
                            .sum::<f64>();
                        reference.u_solve_scale[flat * c * dv + i * dv + d] += gain * lower_term;
                    }
                    for d in 0..dk {
                        let rhs = beta[j] * k[j * dk + d] * cum[j].exp();
                        reference.w_rhs_scale[flat * c * dk + i * dk + d] += gain * rhs.abs();
                        let lower_term = (0..j)
                            .map(|p| {
                                (lower[j * c + p].abs() + lower_magnitude[j * c + p])
                                    * reference.w[flat * c * dk + p * dk + d].abs()
                            })
                            .sum::<f64>();
                        reference.w_solve_scale[flat * c * dk + i * dk + d] += gain * lower_term;
                    }
                }
            }
        }
    }
    reference
}

fn excess(
    actual: &[f32],
    expected: &[f64],
    primary: &[f64],
    secondary: &[f64],
    first_factor: f64,
    second_factor: f64,
) -> f64 {
    actual
        .iter()
        .zip(expected)
        .zip(primary)
        .zip(secondary)
        .map(|(((&a, &e), &p), &s)| {
            let allowed = first_factor * p
                + second_factor * s
                + 8.0 * f32::EPSILON as f64 * e.abs()
                + f32::MIN_POSITIVE as f64;
            (a as f64 - e).abs() / allowed
        })
        .fold(0.0, f64::max)
}

fn check(shape: Shape, input: &Inputs) -> Result<f64, Box<dyn std::error::Error>> {
    let actual = gpu(shape, input)?;
    let expected = oracle(shape, input);
    let ub2 = (1.0 / 256.0f64).powi(2);
    let dot = 3.0 * common::check::gamma_tilde(shape.dk) + ub2;
    let mat = 3.0 * common::check::gamma_tilde(shape.tile) + ub2;
    let normalization = common::check::gamma_tilde(shape.dk)
        + common::check::gamma_tilde(shape.tile)
        + 16.0 * (f32::EPSILON as f64 / 2.0);
    let solve = (2 * (shape.tile / shape.block) + 3) as f64 * mat;
    let u_excess = excess(
        &actual.u,
        &expected.u,
        &expected.u_rhs_scale,
        &expected.u_solve_scale,
        solve,
        dot + normalization,
    );
    let w_excess = excess(
        &actual.w,
        &expected.w,
        &expected.w_rhs_scale,
        &expected.w_solve_scale,
        solve + normalization,
        dot + normalization,
    );
    let intra_excess = excess(
        &actual.intra,
        &expected.intra,
        &expected.intra_scale,
        &expected.intra_decay_scale,
        dot + normalization,
        2.0 * common::check::gamma_tilde(shape.tile),
    );
    let cum_excess = actual
        .cum
        .iter()
        .zip(&expected.cum)
        .map(|(&a, &e)| {
            (a as f64 - e).abs()
                / ((common::check::gamma_tilde(shape.tile) + 8.0 * f32::EPSILON as f64) * e.abs()
                    + f32::MIN_POSITIVE as f64)
        })
        .fold(0.0, f64::max);
    let maximum = u_excess.max(w_excess).max(intra_excess).max(cum_excess);
    if maximum > 1.0 {
        let worst = actual
            .intra
            .iter()
            .zip(&expected.intra)
            .zip(&expected.intra_scale)
            .enumerate()
            .map(|(i, ((&a, &e), &s))| {
                let allowed = (dot + normalization) * s
                    + 8.0 * f32::EPSILON as f64 * e.abs()
                    + f32::MIN_POSITIVE as f64;
                (i, a, e, s, allowed, (a as f64 - e).abs() / allowed)
            })
            .max_by(|a, b| a.5.total_cmp(&b.5))
            .unwrap();
        eprintln!("worst intra {worst:?}");
    }
    assert!(
        maximum <= 1.0,
        "shape={shape:?}, U={u_excess}, W={w_excess}, intra={intra_excess}, cum={cum_excess}"
    );
    Ok(maximum)
}

fn gpu(shape: Shape, x: &Inputs) -> Result<Outputs, Box<dyn std::error::Error>> {
    let stream = common::stream();
    let (n, c, b, kh, vh, dk, dv) = (
        shape.chunks,
        shape.tile,
        shape.block,
        shape.kh,
        shape.vh,
        shape.dk,
        shape.dv,
    );
    let flat = n * vh;
    let upload = |data: &[f32], dims: &[usize]| {
        api::copy_host_vec_to_device(&Arc::new(data.to_vec()))
            .reshape(dims)
            .sync_on(&stream)
    };
    let q = upload(&x.q, &[n, c, kh, dk])?;
    let k = upload(&x.k, &[n, c, kh, dk])?;
    let v = upload(&x.v, &[n, c, vh, dv])?;
    let g = upload(&x.g, &[n, c, vh])?;
    let beta = upload(&x.beta, &[n, c, vh])?;
    let mut ut = api::zeros::<f32>(&[flat, c, c]).sync_on(&stream)?;
    let mut intra = api::zeros::<f32>(&[flat, c, c]).sync_on(&stream)?;
    let mut cum = api::zeros::<f32>(&[flat, c]).sync_on(&stream)?;
    first_phase_matrices(
        (&mut ut).partition([1, c, c]),
        (&mut intra).partition([1, c, c]),
        (&mut cum).partition([1, c]),
        &q,
        &k,
        &g,
        &beta,
        shape.chunk as i32,
        1e-6,
        1.0 / (dk as f32).sqrt(),
    )
    .generics(vec![
        c.to_string(),
        dk.to_string(),
        kh.to_string(),
        vh.to_string(),
    ])
    .sync_on(&stream)?;
    let mut diagonals = api::zeros::<f32>(&[flat * c / b, b, b]).sync_on(&stream)?;
    invert_diagonal((&mut diagonals).partition([1, b, b]), &ut)
        .generics(vec![c.to_string(), b.to_string()])
        .sync_on(&stream)?;
    let mut inverse = api::zeros::<f32>(&[flat, c, c]).sync_on(&stream)?;
    let mut next_inverse = api::zeros::<f32>(&[flat, c, c]).sync_on(&stream)?;
    for row in 0..c / b {
        merge_inverse(
            (&mut next_inverse).partition([1, c, c]),
            &inverse,
            &ut,
            &diagonals,
            row as i32,
        )
        .generics(vec![c.to_string(), b.to_string()])
        .sync_on(&stream)?;
        std::mem::swap(&mut inverse, &mut next_inverse);
    }
    let mut u = api::zeros::<f32>(&[flat, c, dv]).sync_on(&stream)?;
    let mut w = api::zeros::<f32>(&[flat, c, dk]).sync_on(&stream)?;
    apply_inverse(
        (&mut u).partition([1, c, dv]),
        (&mut w).partition([1, c, dk]),
        &inverse,
        &cum,
        &k,
        &v,
        &beta,
        shape.chunk as i32,
        1e-6,
    )
    .generics(vec![
        c.to_string(),
        dk.to_string(),
        dv.to_string(),
        kh.to_string(),
        vh.to_string(),
    ])
    .sync_on(&stream)?;
    Ok(Outputs {
        u: u.to_host_vec().sync_on(&stream)?,
        w: w.to_host_vec().sync_on(&stream)?,
        intra: intra.to_host_vec().sync_on(&stream)?,
        cum: cum.to_host_vec().sync_on(&stream)?,
    })
}

fn second_gpu(
    shape: Shape,
    x: &Inputs,
    first: &Outputs,
    initial: &[f32],
    value_tile: usize,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let (n, c, kh, vh, dk, dv) = (
        shape.chunks,
        shape.tile,
        shape.kh,
        shape.vh,
        shape.dk,
        shape.dv,
    );
    let upload = |data: &[f32], dims: &[usize]| {
        api::copy_host_vec_to_device(&Arc::new(data.to_vec()))
            .reshape(dims)
            .sync_on(&stream)
    };
    let q = upload(&x.q, &[n, c, kh, dk])?;
    let k = upload(&x.k, &[n, c, kh, dk])?;
    let u = upload(&first.u, &[n * vh, c, dv])?;
    let w = upload(&first.w, &[n * vh, c, dk])?;
    let intra = upload(&first.intra, &[n * vh, c, c])?;
    let cum = upload(&first.cum, &[n * vh, c])?;
    let mut state = upload(initial, &[1, vh * dk, dv])?;
    let mut out = api::zeros::<f32>(&[n * c, vh, dv]).sync_on(&stream)?;
    second_phase(
        (&mut out).partition([(n * c).next_power_of_two(), 1, value_tile]),
        state.partition([1, dk, value_tile]),
        &u,
        &w,
        &intra,
        &cum,
        &q,
        &k,
        n as i32,
        1e-6,
        1.0 / (dk as f32).sqrt(),
    )
    .generics(vec![
        (n * c).next_power_of_two().to_string(),
        c.to_string(),
        dk.to_string(),
        value_tile.to_string(),
        kh.to_string(),
        vh.to_string(),
    ])
    .sync_on(&stream)?;
    Ok((
        out.to_host_vec().sync_on(&stream)?,
        state.to_host_vec().sync_on(&stream)?,
    ))
}

#[test]
fn second_phase_smoke() -> Result<(), Box<dyn std::error::Error>> {
    let shape = Shape {
        chunks: 2,
        chunk: 5,
        tile: 16,
        block: 16,
        kh: 1,
        vh: 2,
        dk: 32,
        dv: 32,
    };
    let input = generated(shape, 9, 24, false);
    let first = gpu(shape, &input)?;
    let initial = vec![0.0; shape.vh * shape.dk * shape.dv];
    let (out, state) = second_gpu(shape, &input, &first, &initial, 16)?;
    assert!(out.iter().all(|x| x.is_finite()));
    assert!(state.iter().any(|&x| x != 0.0));
    Ok(())
}

struct SecondReference {
    output: Vec<f64>,
    state: Vec<f64>,
    output_bound: Vec<f64>,
    state_bound: Vec<f64>,
}

fn second_oracle_with_decay(
    shape: Shape,
    x: &Inputs,
    first: &Outputs,
    initial: &[f32],
    omit_decay: bool,
) -> SecondReference {
    let (n, c, kh, vh, dk, dv) = (
        shape.chunks,
        shape.tile,
        shape.kh,
        shape.vh,
        shape.dk,
        shape.dv,
    );
    let unit = f32::EPSILON as f64 / 2.0;
    let split = (1.0 / 256.0f64).powi(2);
    let norm = common::check::gamma_tilde(dk) + 16.0 * unit;
    let key_dot = 3.0 * common::check::gamma_tilde(c) + 3.0 * split + norm + 8.0 * unit;
    let query_dot = 3.0 * common::check::gamma_tilde(dk) + 3.0 * split + norm + 8.0 * unit;
    let w_dot = 3.0 * common::check::gamma_tilde(dk) + 3.0 * split;
    let intra_dot = 3.0 * common::check::gamma_tilde(c) + 3.0 * split;
    let mut result = SecondReference {
        output: vec![0.0; n * c * vh * dv],
        state: initial.iter().map(|&v| v as f64).collect(),
        output_bound: vec![0.0; n * c * vh * dv],
        state_bound: vec![0.0; vh * dk * dv],
    };
    for chunk in 0..n {
        for head in 0..vh {
            let flat = chunk * vh + head;
            let key_head = head / (vh / kh);
            let last = first.cum[flat * c + c - 1] as f64;
            let decay = if omit_decay { 1.0 } else { last.exp() };
            let mut q = vec![0.0f64; c * dk];
            let mut k = vec![0.0f64; c * dk];
            for row in 0..c {
                let time = chunk * c + row;
                let base = (time * kh + key_head) * dk;
                let qnorm =
                    ((0..dk).map(|d| (x.q[base + d] as f64).powi(2)).sum::<f64>() + 1e-6).sqrt();
                let knorm =
                    ((0..dk).map(|d| (x.k[base + d] as f64).powi(2)).sum::<f64>() + 1e-6).sqrt();
                let cum = first.cum[flat * c + row] as f64;
                for d in 0..dk {
                    q[row * dk + d] = x.q[base + d] as f64 / qnorm / (dk as f64).sqrt() * cum.exp();
                    k[row * dk + d] = x.k[base + d] as f64 / knorm * (last - cum).exp();
                }
            }
            let mut delta = vec![0.0f64; c * dv];
            let mut delta_bound = vec![0.0f64; c * dv];
            for row in 0..c {
                for d in 0..dv {
                    let mut dot = 0.0;
                    let mut magnitude = 0.0;
                    let mut propagated = 0.0;
                    for j in 0..dk {
                        let weight = first.w[(flat * c + row) * dk + j] as f64;
                        let state_index = (head * dk + j) * dv + d;
                        dot += weight * result.state[state_index];
                        magnitude += (weight * result.state[state_index]).abs();
                        propagated += weight.abs() * result.state_bound[state_index];
                    }
                    let index = row * dv + d;
                    delta[index] = first.u[(flat * c + row) * dv + d] as f64 - dot;
                    delta_bound[index] =
                        w_dot * magnitude + propagated + 8.0 * unit * delta[index].abs();
                }
            }
            for row in 0..c {
                for d in 0..dv {
                    let mut inter = 0.0;
                    let mut inter_magnitude = 0.0;
                    let mut inter_propagated = 0.0;
                    for j in 0..dk {
                        let state_index = (head * dk + j) * dv + d;
                        let product = q[row * dk + j] * result.state[state_index];
                        inter += product;
                        inter_magnitude += product.abs();
                        inter_propagated += q[row * dk + j].abs() * result.state_bound[state_index];
                    }
                    let mut within = 0.0;
                    let mut within_magnitude = 0.0;
                    let mut within_propagated = 0.0;
                    for j in 0..c {
                        let a = first.intra[(flat * c + row) * c + j] as f64;
                        let product = a * delta[j * dv + d];
                        within += product;
                        within_magnitude += product.abs();
                        within_propagated += a.abs() * delta_bound[j * dv + d];
                    }
                    let index = ((chunk * c + row) * vh + head) * dv + d;
                    result.output[index] = inter + within;
                    result.output_bound[index] = query_dot * inter_magnitude
                        + intra_dot * within_magnitude
                        + inter_propagated
                        + within_propagated
                        + 8.0 * unit * result.output[index].abs();
                }
            }
            for j in 0..dk {
                for d in 0..dv {
                    let index = (head * dk + j) * dv + d;
                    let mut update = 0.0;
                    let mut magnitude = 0.0;
                    let mut propagated = 0.0;
                    for row in 0..c {
                        let weight = k[row * dk + j];
                        let product = weight * delta[row * dv + d];
                        update += product;
                        magnitude += product.abs();
                        propagated += weight.abs() * delta_bound[row * dv + d];
                    }
                    let next = decay * result.state[index] + update;
                    result.state_bound[index] = decay * result.state_bound[index]
                        + 8.0 * unit * decay * result.state[index].abs()
                        + key_dot * magnitude
                        + propagated
                        + 8.0 * unit * next.abs();
                    result.state[index] = next;
                }
            }
        }
    }
    result
}

fn second_oracle(shape: Shape, x: &Inputs, first: &Outputs, initial: &[f32]) -> SecondReference {
    second_oracle_with_decay(shape, x, first, initial, false)
}

fn second_excess(actual: &[f32], expected: &[f64], bound: &[f64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .zip(bound)
        .map(|((&a, &e), &b)| (a as f64 - e).abs() / b.max(f64::MIN_POSITIVE))
        .fold(0.0, f64::max)
}

#[test]
fn second_phase_oracle_pbt() -> Result<(), Box<dyn std::error::Error>> {
    let mut runner = common::runner(64, file!());
    runner.run(
        &(
            1usize..97,
            proptest::sample::select(vec![5usize, 16, 31, 64]),
            1usize..3,
            any::<u64>(),
            any::<bool>(),
            any::<bool>(),
        ),
        |(time, chunk, kh, seed, strong, nonzero)| {
            let shape = Shape {
                chunks: time.div_ceil(chunk),
                chunk,
                tile: chunk.next_power_of_two().max(16),
                block: 16,
                kh,
                vh: kh * 2,
                dk: 32,
                dv: 32,
            };
            let input = generated(shape, time, seed, strong);
            let first = gpu(shape, &input).map_err(|e| TestCaseError::fail(e.to_string()))?;
            let mut initial = vec![0.0f32; shape.vh * shape.dk * shape.dv];
            if nonzero {
                for (i, value) in initial.iter_mut().enumerate() {
                    *value = ((i * 17 % 61) as f32 - 30.0) / 97.0;
                }
            }
            let expected = second_oracle(shape, &input, &first, &initial);
            let (actual_output, actual_state) = second_gpu(shape, &input, &first, &initial, 16)
                .map_err(|e| TestCaseError::fail(e.to_string()))?;
            let output = second_excess(&actual_output, &expected.output, &expected.output_bound);
            let state = second_excess(&actual_state, &expected.state, &expected.state_bound);
            prop_assert!(
                output <= 1.0 && state <= 1.0,
                "output={output} state={state}"
            );
            Ok(())
        },
    )?;
    Ok(())
}

fn decode_gpu(
    shape: Shape,
    input: &Inputs,
    initial: &[f32],
    time: usize,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let (kh, vh, dk, dv) = (shape.kh, shape.vh, shape.dk, shape.dv);
    let upload = |data: &[f32], dims: &[usize]| {
        api::copy_host_vec_to_device(&Arc::new(data.to_vec()))
            .reshape(dims)
            .sync_on(&stream)
    };
    let mut state = upload(initial, &[vh, dk, dv])?;
    let zeros = vec![0.0; vh];
    let a_log = upload(&zeros, &[vh])?;
    let dt_bias = upload(&zeros, &[vh])?;
    let mut result = vec![0.0; time * vh * dv];
    for t in 0..time {
        let mut output = api::zeros::<f32>(&[vh, 1, dv]).sync_on(&stream)?;
        let row = t / shape.chunk * shape.tile + t % shape.chunk;
        let q = upload(&input.q[row * kh * dk..(row + 1) * kh * dk], &[kh, dk])?;
        let k = upload(&input.k[row * kh * dk..(row + 1) * kh * dk], &[kh, dk])?;
        let v = upload(&input.v[row * vh * dv..(row + 1) * vh * dv], &[vh, dv])?;
        let b: Vec<f32> = input.beta[row * vh..(row + 1) * vh]
            .iter()
            .map(|&beta| (beta / (1.0 - beta)).ln())
            .collect();
        let a: Vec<f32> = input.g[row * vh..(row + 1) * vh]
            .iter()
            .map(|&g| (-g).exp_m1().ln())
            .collect();
        let b = upload(&b, &[vh])?;
        let a = upload(&a, &[vh])?;
        delta_step(
            (&mut output).partition([1, 1, 16]),
            state.partition([1, dk, 16]),
            &q,
            &k,
            &v,
            &b,
            &a,
            &a_log,
            &dt_bias,
            1e-6,
            1.0 / (dk as f32).sqrt(),
        )
        .generics(vec![
            kh.to_string(),
            vh.to_string(),
            dk.to_string(),
            "16".into(),
        ])
        .sync_on(&stream)?;
        let step: Vec<f32> = output.to_host_vec().sync_on(&stream)?;
        result[t * vh * dv..(t + 1) * vh * dv].copy_from_slice(&step);
    }
    Ok((result, state.to_host_vec().sync_on(&stream)?))
}

#[test]
fn second_phase_matches_delta_step() -> Result<(), Box<dyn std::error::Error>> {
    for (chunk, time, seed, nonzero) in [(5usize, 13usize, 91, false), (16, 31, 719, true)] {
        let shape = Shape {
            chunks: time.div_ceil(chunk),
            chunk,
            tile: chunk.next_power_of_two().max(16),
            block: 16,
            kh: 2,
            vh: 4,
            dk: 32,
            dv: 32,
        };
        let input = generated(shape, time, seed, false);
        let first = gpu(shape, &input)?;
        let mut initial = vec![0.0; shape.vh * shape.dk * shape.dv];
        if nonzero {
            for (i, value) in initial.iter_mut().enumerate() {
                *value = ((i * 13 % 29) as f32 - 14.0) / 89.0;
            }
        }
        let oracle = second_oracle(shape, &input, &first, &initial);
        let (chunk_output, chunk_state) = second_gpu(shape, &input, &first, &initial, 16)?;
        let (decode_output, decode_state) = decode_gpu(shape, &input, &initial, time)?;
        let mut worst = 0.0f64;
        for t in 0..time {
            for h in 0..shape.vh {
                for d in 0..shape.dv {
                    let padded_time = t / shape.chunk * shape.tile + t % shape.chunk;
                    let padded = (padded_time * shape.vh + h) * shape.dv + d;
                    let dense = (t * shape.vh + h) * shape.dv + d;
                    let allowance =
                        oracle.output_bound[padded] + 3e-4 * (1.0 + oracle.output[padded].abs());
                    let error = (chunk_output[padded] - decode_output[dense]).abs() as f64;
                    worst = worst.max(error / allowance);
                }
            }
        }
        for ((&a, &b), (&e, &bound)) in chunk_state
            .iter()
            .zip(&decode_state)
            .zip(oracle.state.iter().zip(&oracle.state_bound))
        {
            let allowance = bound + 3e-4 * (1.0 + e.abs());
            worst = worst.max((a - b).abs() as f64 / allowance);
        }
        assert!(worst <= 1.0, "chunk={chunk} time={time} excess={worst}");
    }
    Ok(())
}

#[test]
fn second_phase_omissions_are_detected() -> Result<(), Box<dyn std::error::Error>> {
    let shape = Shape {
        chunks: 3,
        chunk: 16,
        tile: 16,
        block: 16,
        kh: 1,
        vh: 2,
        dk: 32,
        dv: 32,
    };
    let input = generated(shape, 48, 9217, false);
    let first = gpu(shape, &input)?;
    let initial: Vec<f32> = (0..shape.vh * shape.dk * shape.dv)
        .map(|i| ((i * 17 % 41) as f32 - 20.0) / 53.0)
        .collect();
    let expected = second_oracle(shape, &input, &first, &initial);
    let omitted_decay = second_oracle_with_decay(shape, &input, &first, &initial, true);
    let omitted_w = Outputs {
        u: first.u.clone(),
        w: vec![0.0; first.w.len()],
        intra: first.intra.clone(),
        cum: first.cum.clone(),
    };
    let (wrong_output, wrong_state) = second_gpu(shape, &input, &omitted_w, &initial, 16)?;
    assert!(second_excess(&wrong_output, &expected.output, &expected.output_bound) > 1.0);
    assert!(second_excess(&wrong_state, &expected.state, &expected.state_bound) > 1.0);
    assert!(
        second_excess(
            &omitted_decay
                .output
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<_>>(),
            &expected.output,
            &expected.output_bound,
        ) > 1.0
    );
    assert!(
        second_excess(
            &omitted_decay
                .state
                .iter()
                .map(|&x| x as f32)
                .collect::<Vec<_>>(),
            &expected.state,
            &expected.state_bound,
        ) > 1.0
    );
    Ok(())
}

#[test]
fn smoke() -> Result<(), Box<dyn std::error::Error>> {
    let shape = Shape {
        chunks: 1,
        chunk: 32,
        tile: 32,
        block: 16,
        kh: 2,
        vh: 4,
        dk: 32,
        dv: 32,
    };
    let mut q = vec![0.0; shape.chunks * shape.tile * shape.kh * shape.dk];
    let mut k = q.clone();
    let mut v = vec![0.0; shape.chunks * shape.tile * shape.vh * shape.dv];
    for (i, x) in q.iter_mut().enumerate() {
        *x = ((i * 17 % 103) as f32 - 51.0) / 101.0;
    }
    for (i, x) in k.iter_mut().enumerate() {
        *x = ((i * 19 % 107) as f32 - 53.0) / 101.0;
    }
    for (i, x) in v.iter_mut().enumerate() {
        *x = ((i * 23 % 109) as f32 - 54.0) / 101.0;
    }
    let g = vec![-0.03; shape.chunks * shape.tile * shape.vh];
    let beta = vec![0.4; shape.chunks * shape.tile * shape.vh];
    check(shape, &Inputs { q, k, v, g, beta })?;
    Ok(())
}

fn generated(shape: Shape, time: usize, seed: u64, strong: bool) -> Inputs {
    let mut state = seed.max(1);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u32 << 24) as f32
    };
    let (n, c, kh, vh, dk, dv) = (
        shape.chunks,
        shape.tile,
        shape.kh,
        shape.vh,
        shape.dk,
        shape.dv,
    );
    let mut input = Inputs {
        q: vec![0.0; n * c * kh * dk],
        k: vec![0.0; n * c * kh * dk],
        v: vec![0.0; n * c * vh * dv],
        g: vec![0.0; n * c * vh],
        beta: vec![0.0; n * c * vh],
    };
    for t in 0..time {
        let row = t / shape.chunk * c + t % shape.chunk;
        for x in &mut input.q[row * kh * dk..(row + 1) * kh * dk] {
            *x = (next() * 2.0 - 1.0) * if next() < 0.1 { 3.0 } else { 1.0 };
        }
        for x in &mut input.k[row * kh * dk..(row + 1) * kh * dk] {
            *x = (next() * 2.0 - 1.0) * if next() < 0.1 { 3.0 } else { 1.0 };
        }
        for x in &mut input.v[row * vh * dv..(row + 1) * vh * dv] {
            *x = next() * 2.0 - 1.0;
        }
        for h in 0..vh {
            input.beta[row * vh + h] = 0.01 + 0.98 * next();
            input.g[row * vh + h] = if strong {
                -0.5 - 5.0 * next()
            } else {
                -0.001 - 0.1 * next()
            };
        }
    }
    input
}

#[test]
fn oracle_pbt() -> Result<(), Box<dyn std::error::Error>> {
    let mut runner = common::runner(64, file!());
    runner.run(
        &(
            1usize..129,
            proptest::sample::select(vec![5usize, 16, 31, 64]),
            1usize..3,
            any::<u64>(),
            any::<bool>(),
        ),
        |(time, chunk, kh, seed, strong)| {
            let shape = Shape {
                chunks: time.div_ceil(chunk),
                chunk,
                tile: chunk.next_power_of_two().max(16),
                block: 16,
                kh,
                vh: kh * 2,
                dk: 32,
                dv: 32,
            };
            let input = generated(shape, time, seed, strong);
            check(shape, &input)
                .map(|_| ())
                .map_err(|e| TestCaseError::fail(e.to_string()))
        },
    )?;
    Ok(())
}

#[test]
fn model_shape_oracle() -> Result<(), Box<dyn std::error::Error>> {
    let shape = Shape {
        chunks: 1,
        chunk: 64,
        tile: 64,
        block: 16,
        kh: 16,
        vh: 32,
        dk: 128,
        dv: 128,
    };
    check(shape, &generated(shape, 64, 17, false))?;
    Ok(())
}

#[test]
#[ignore = "needs refs/qwen36-prefix4-prefill.safetensors"]
fn hf_prefill_inputs() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../refs/qwen36-prefix4-prefill.safetensors");
    let bytes = std::fs::read(path)?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
    for layer in 0..3 {
        let prefix = format!("layer{layer}/gdn/chunk");
        let read = |name: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let tensor = tensors.tensor(&format!("{prefix}/{name}"))?;
            assert_eq!(tensor.dtype(), safetensors::Dtype::F32);
            Ok(tensor
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect())
        };
        let q = read("q")?;
        let k = read("k")?;
        let v = read("v")?;
        let g = read("g")?;
        let beta = read("beta")?;
        let time = g.len() / 32;
        let shape = Shape {
            chunks: time.div_ceil(64),
            chunk: 64,
            tile: 64,
            block: 16,
            kh: 32,
            vh: 32,
            dk: q.len() / (time * 32),
            dv: v.len() / (time * 32),
        };
        let pack = |src: &[f32], width: usize| {
            let mut out = vec![0.0; shape.chunks * shape.tile * width];
            out[..src.len()].copy_from_slice(src);
            out
        };
        let input = Inputs {
            q: pack(&q, shape.kh * shape.dk),
            k: pack(&k, shape.kh * shape.dk),
            v: pack(&v, shape.vh * shape.dv),
            g: pack(&g, shape.vh),
            beta: pack(&beta, shape.vh),
        };
        let ratio = check(shape, &input)?;
        eprintln!("HF layer {layer}: max bound ratio {ratio:.3}");
    }
    Ok(())
}

fn normalized_rows(shape: Shape, input: &Inputs, head: usize, time: usize) -> (Vec<f64>, Vec<f64>) {
    let key_head = head / (shape.vh / shape.kh);
    let row = time / shape.chunk * shape.tile + time % shape.chunk;
    let base = (row * shape.kh + key_head) * shape.dk;
    let normalize = |src: &[f32], query: bool| {
        let raw = &src[base..base + shape.dk];
        let norm = (raw.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() + 1e-6).sqrt();
        raw.iter()
            .map(|&x| x as f64 / norm / if query { (shape.dk as f64).sqrt() } else { 1.0 })
            .collect::<Vec<_>>()
    };
    (normalize(&input.q, true), normalize(&input.k, false))
}

#[test]
fn second_phase_matches_decode() -> Result<(), Box<dyn std::error::Error>> {
    let shape = Shape {
        chunks: 3,
        chunk: 5,
        tile: 16,
        block: 16,
        kh: 2,
        vh: 4,
        dk: 32,
        dv: 32,
    };
    let time = 13;
    let input = generated(shape, time, 1945, false);
    let actual = gpu(shape, &input)?;
    for head in 0..shape.vh {
        let mut chunk_state = vec![0.0f64; shape.dk * shape.dv];
        let mut decode_state = chunk_state.clone();
        for chunk in 0..shape.chunks {
            let flat = chunk * shape.vh + head;
            let start_state = chunk_state.clone();
            let mut v_new = vec![0.0f64; shape.chunk * shape.dv];
            for row in 0..shape.chunk {
                for d in 0..shape.dv {
                    v_new[row * shape.dv + d] = actual.u[(flat * shape.tile + row) * shape.dv + d]
                        as f64
                        - (0..shape.dk)
                            .map(|k| {
                                actual.w[(flat * shape.tile + row) * shape.dk + k] as f64
                                    * start_state[k * shape.dv + d]
                            })
                            .sum::<f64>();
                }
            }
            let last = actual.cum[flat * shape.tile + shape.chunk - 1] as f64;
            for k in 0..shape.dk {
                for d in 0..shape.dv {
                    chunk_state[k * shape.dv + d] *= last.exp();
                }
            }
            for row in 0..shape.chunk {
                let t = chunk * shape.chunk + row;
                if t >= time {
                    break;
                }
                let (q, k) = normalized_rows(shape, &input, head, t);
                let cum = actual.cum[flat * shape.tile + row] as f64;
                let g = input.g[(chunk * shape.tile + row) * shape.vh + head] as f64;
                let beta = input.beta[(chunk * shape.tile + row) * shape.vh + head] as f64;
                for d in 0..shape.dv {
                    let inter = (0..shape.dk)
                        .map(|j| q[j] * start_state[j * shape.dv + d])
                        .sum::<f64>()
                        * cum.exp();
                    let intra = (0..=row)
                        .map(|j| {
                            actual.intra[(flat * shape.tile + row) * shape.tile + j] as f64
                                * v_new[j * shape.dv + d]
                        })
                        .sum::<f64>();
                    let chunk_output = inter + intra;
                    let old = (0..shape.dk)
                        .map(|j| k[j] * decode_state[j * shape.dv + d])
                        .sum::<f64>()
                        * g.exp();
                    let delta = beta
                        * (input.v[((chunk * shape.tile + row) * shape.vh + head) * shape.dv + d]
                            as f64
                            - old);
                    for j in 0..shape.dk {
                        decode_state[j * shape.dv + d] =
                            g.exp() * decode_state[j * shape.dv + d] + k[j] * delta;
                    }
                    let decode_output = (0..shape.dk)
                        .map(|j| q[j] * decode_state[j * shape.dv + d])
                        .sum::<f64>();
                    assert!(
                        (chunk_output - decode_output).abs() <= 5e-4 * (1.0 + decode_output.abs()),
                        "head={head}, t={t}, d={d}"
                    );
                }
                let weight = (last - cum).exp();
                for j in 0..shape.dk {
                    for d in 0..shape.dv {
                        chunk_state[j * shape.dv + d] += weight * k[j] * v_new[row * shape.dv + d];
                    }
                }
            }
        }
        for (a, b) in chunk_state.iter().zip(&decode_state) {
            assert!((a - b).abs() <= 5e-4 * (1.0 + b.abs()));
        }
    }
    Ok(())
}

#[test]
fn omitted_decay_and_solve_are_detected() {
    let shape = Shape {
        chunks: 1,
        chunk: 16,
        tile: 16,
        block: 16,
        kh: 1,
        vh: 2,
        dk: 32,
        dv: 32,
    };
    let input = generated(shape, 16, 991, false);
    let expected = oracle(shape, &input);
    let mut omitted_solve = vec![0.0f32; expected.u.len()];
    let mut omitted_decay = vec![0.0f32; expected.intra.len()];
    for head in 0..shape.vh {
        for row in 0..shape.chunk {
            let t = row;
            let (q, _) = normalized_rows(shape, &input, head, t);
            let beta = input.beta[t * shape.vh + head];
            for d in 0..shape.dv {
                omitted_solve[(head * shape.tile + row) * shape.dv + d] =
                    beta * input.v[(t * shape.vh + head) * shape.dv + d];
            }
            for col in 0..=row {
                let (_, k) = normalized_rows(shape, &input, head, col);
                omitted_decay[(head * shape.tile + row) * shape.tile + col] =
                    q.iter().zip(k).map(|(a, b)| a * b).sum::<f64>() as f32;
            }
        }
    }
    let zero_intra = vec![0.0; expected.intra.len()];
    assert!(
        excess(
            &omitted_solve,
            &expected.u,
            &expected.u_rhs_scale,
            &expected.u_solve_scale,
            3.0 * common::check::gamma_tilde(shape.tile),
            3.0 * common::check::gamma_tilde(shape.dk)
        ) > 1.0
    );
    assert!(
        excess(
            &omitted_decay,
            &expected.intra,
            &expected.intra_scale,
            &zero_intra,
            3.0 * common::check::gamma_tilde(shape.dk),
            0.0
        ) > 1.0
    );
}

#[test]
#[ignore = "release GPU timing on RTX 5080"]
fn release_timing() -> Result<(), Box<dyn std::error::Error>> {
    cutile::jit_cache::enable_default()?;
    for time in [512usize, 2048, 5000] {
        let shape = Shape {
            chunks: time.div_ceil(64),
            chunk: 64,
            tile: 64,
            block: 16,
            kh: 16,
            vh: 32,
            dk: 128,
            dv: 128,
        };
        let input = generated(shape, time, 41, false);
        gpu(shape, &input)?;
        let mut milliseconds = Vec::new();
        for _ in 0..3 {
            let start = std::time::Instant::now();
            gpu(shape, &input)?;
            milliseconds.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        milliseconds.sort_by(f64::total_cmp);
        eprintln!(
            "T={time}, Hv={}, median end-to-end ms={:.3}, samples={milliseconds:?}",
            shape.vh, milliseconds[1]
        );
    }
    Ok(())
}
