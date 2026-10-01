mod common;

use common::{check, strategy};
use cuda_async::device_operation::DeviceOp;
use cuda_core::Stream;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, ToHostVec};
use cutile::tile_kernel::TileKernel;
use firepit_kernels::rope::{RopeDims, frequencies, generics, rope};
use proptest::prelude::*;
use std::sync::Arc;

fn gpu(
    stream: &Arc<Stream>,
    input: &[f32],
    positions: &[usize],
    dims: &RopeDims,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let (heads, head_dim, rotary_dim) = (dims.heads(), dims.head_dim(), dims.rotary_dim());
    let batch = positions.len();
    assert_eq!(input.len(), batch * heads * head_dim);
    let (cos, sin) = frequencies(positions, dims);
    let x = api::copy_host_vec_to_device(&Arc::new(input.to_vec()))
        .reshape(&[batch * heads, head_dim])
        .sync_on(stream)?;
    let c = api::copy_host_vec_to_device(&Arc::new(cos))
        .reshape(&[batch, head_dim])
        .sync_on(stream)?;
    let s = api::copy_host_vec_to_device(&Arc::new(sin))
        .reshape(&[batch, head_dim])
        .sync_on(stream)?;
    let mut output = api::zeros::<f32>(&[batch * heads, head_dim]).sync_on(stream)?;
    rope((&mut output).partition([1, rotary_dim]), &x, &c, &s)
        .generics(generics(dims))
        .sync_on(stream)?;
    Ok(output.to_host_vec().sync_on(stream)?)
}

fn oracle(input: &[f32], positions: &[usize], dims: &RopeDims) -> Vec<f64> {
    let (heads, head_dim, rotary_dim) = (dims.heads(), dims.head_dim(), dims.rotary_dim());
    let (cos, sin) = frequencies(positions, dims);
    let half = rotary_dim / 2;
    input
        .chunks_exact(head_dim)
        .enumerate()
        .flat_map(|(row, x)| {
            let offset = row / heads * head_dim;
            (0..head_dim)
                .map(|i| {
                    if i >= rotary_dim {
                        x[i] as f64
                    } else {
                        let other = if i < half {
                            x[i + half] as f64
                        } else {
                            x[i - half] as f64
                        };
                        x[i] as f64 * cos[offset + i] as f64
                            + if i < half { -other } else { other } * sin[offset + i] as f64
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
}

#[test]
fn rope_matches_f64_oracle() {
    let stream = common::stream();
    common::runner(64, file!())
        .run(
            &(
                prop::sample::select(vec![1usize, 2, 4]),
                prop::sample::select(vec![1usize, 2, 16]),
                prop::sample::select(vec![(256usize, 64usize), (128, 64), (128, 128)]),
            )
                .prop_flat_map(|(batch, heads, (head_dim, rotary_dim))| {
                    (
                        Just((heads, head_dim, rotary_dim)),
                        prop::collection::vec(0usize..=262143, batch),
                        strategy::values(batch * heads * head_dim),
                    )
                }),
            |((heads, head_dim, rotary_dim), positions, input)| {
                let dims = RopeDims::new(heads, head_dim, rotary_dim, 10_000_000.0).unwrap();
                let actual = gpu(&stream, &input, &positions, &dims)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let expected = oracle(&input, &positions, &dims);
                prop_assert!(check::max_relative(&actual, &expected) < 3e-6);
                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn rope_preserves_norm_and_relative_position() -> Result<(), Box<dyn std::error::Error>> {
    let stream = common::stream();
    let q: Vec<f32> = (0..256)
        .map(|i| ((i * 17 % 61) as f32 - 30.0) / 31.0)
        .collect();
    let k: Vec<f32> = (0..256)
        .map(|i| ((i * 29 % 67) as f32 - 33.0) / 34.0)
        .collect();
    let mut input = Vec::new();
    for row in [&q, &k, &q, &k, &q, &k] {
        input.extend(row);
    }
    let dims = RopeDims::new(1, 256, 64, 10_000_000.0)?;
    let result = gpu(&stream, &input, &[0, 100, 200, 300, 4095, 4195], &dims)?;
    assert!(check::max_abs(&result[..256], &q) < 1e-7);
    for row in result.chunks_exact(256).zip(input.chunks_exact(256)) {
        assert!((dot(row.0, row.0) - dot(row.1, row.1)).abs() < 1e-4);
    }
    let d1 = dot(&result[2 * 256..3 * 256], &result[3 * 256..4 * 256]);
    let d2 = dot(&result[4 * 256..5 * 256], &result[5 * 256..6 * 256]);
    assert!((d1 - d2).abs() < 1e-4);
    Ok(())
}
