use cuda_async::device_operation::DeviceOp;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, Tensor, ToHostVec};
use cutile::tile_kernel::TileKernel;
use firepit::Checkpoint;
use firepit_kernels::gemv::DEFAULT_GEMV_TILE;
use firepit_kernels::gemv::gemv;
use firepit_kernels::moe::MoeDims;
use firepit_kernels::swiglu::swiglu;
use firepit_kernels::weights::{DenseMatrix, Weight};
use half::bf16;
use std::sync::Arc;

pub(super) struct Expert {
    pub(super) gate_up: DenseMatrix<bf16>,
    pub(super) down: DenseMatrix<bf16>,
    pub(super) gate_up_host: Vec<bf16>,
    pub(super) down_host: Vec<bf16>,
}

pub(super) fn load_expert(
    checkpoint: &Checkpoint,
    layer: usize,
    stream: &Arc<cuda_core::Stream>,
    dims: MoeDims,
) -> Result<Expert, Box<dyn std::error::Error>> {
    let name = |projection| {
        format!("model.language_model.layers.{layer}.mlp.shared_expert.{projection}_proj.weight")
    };
    let gate = checkpoint.tensor_bf16(&name("gate"))?;
    let up = checkpoint.tensor_bf16(&name("up"))?;
    let down = checkpoint.tensor_bf16(&name("down"))?;
    assert_eq!(gate.shape, [dims.shared_intermediate(), dims.hidden()]);
    assert_eq!(up.shape, [dims.shared_intermediate(), dims.hidden()]);
    assert_eq!(down.shape, [dims.hidden(), dims.shared_intermediate()]);
    let mut gate_up_host = gate.data;
    gate_up_host.extend(up.data);
    let gate_up = DenseMatrix::upload(
        gate_up_host.clone(),
        2 * dims.shared_intermediate(),
        dims.hidden(),
        stream,
    )?;
    let down_matrix = DenseMatrix::upload(
        down.data.clone(),
        dims.hidden(),
        dims.shared_intermediate(),
        stream,
    )?;
    Ok(Expert {
        gate_up,
        down: down_matrix,
        gate_up_host,
        down_host: down.data,
    })
}

pub(super) fn gpu_expert(
    stream: &Arc<cuda_core::Stream>,
    gate_up: &DenseMatrix<bf16>,
    down: &DenseMatrix<bf16>,
    input: &[f32],
    dims: MoeDims,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let x = api::copy_host_vec_to_device(&Arc::new(input.to_vec()))
        .reshape(&[1, dims.hidden()])
        .sync_on(stream)?;
    Ok(gpu_expert_tensor(stream, gate_up, down, &x, dims)?
        .to_host_vec()
        .sync_on(stream)?)
}

pub(super) fn gpu_expert_tensor(
    stream: &Arc<cuda_core::Stream>,
    gate_up: &DenseMatrix<bf16>,
    down: &DenseMatrix<bf16>,
    x: &Tensor<f32>,
    dims: MoeDims,
) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
    let mut projected = api::zeros::<f32>(&[1, 2 * dims.shared_intermediate()]).sync_on(stream)?;
    gemv(
        &Weight::Dense(gate_up.clone()),
        &x.view(&[1, dims.hidden() / 32, 32])?,
        (&mut projected).partition([1, DEFAULT_GEMV_TILE.rows]),
        DEFAULT_GEMV_TILE,
    )
    .sync_on(stream)?;
    let mut activated = api::zeros::<f32>(&[1, dims.shared_intermediate()]).sync_on(stream)?;
    swiglu(
        (&mut activated).partition([1, dims.shared_intermediate()]),
        &projected,
    )
    .generics(vec![dims.shared_intermediate().to_string()])
    .sync_on(stream)?;
    let mut output = api::zeros::<f32>(&[1, dims.hidden()]).sync_on(stream)?;
    gemv(
        &Weight::Dense(down.clone()),
        &activated.view(&[1, dims.shared_intermediate() / 32, 32])?,
        (&mut output).partition([1, DEFAULT_GEMV_TILE.rows]),
        DEFAULT_GEMV_TILE,
    )
    .sync_on(stream)?;
    Ok(output)
}

pub(super) fn cpu_expert(input: &[f32], expert: &Expert, dims: MoeDims) -> Vec<f64> {
    let projected: Vec<f64> = expert
        .gate_up_host
        .chunks_exact(dims.hidden())
        .map(|row| {
            row.iter()
                .zip(input)
                .map(|(&w, &x)| w.to_f32() as f64 * x as f64)
                .sum()
        })
        .collect();
    let activated: Vec<f64> = projected[..dims.shared_intermediate()]
        .iter()
        .zip(&projected[dims.shared_intermediate()..])
        .map(|(&gate, &up)| gate / (1.0 + (-gate).exp()) * up)
        .collect();
    expert
        .down_host
        .chunks_exact(dims.shared_intermediate())
        .map(|row| {
            row.iter()
                .zip(&activated)
                .map(|(&w, &x)| w.to_f32() as f64 * x)
                .sum()
        })
        .collect()
}
