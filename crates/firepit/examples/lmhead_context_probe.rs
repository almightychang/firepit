use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::PartitionMut;
use firepit::{Engine, LoadOptions, Model, SessionId};
use firepit_kernels::gemv::{GemvTile, gemv};
use firepit_kernels::sink::{Eager, Graph};
use firepit_kernels::weights::{DenseMatrix, Weight};
use half::bf16;
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// lm_head inside graph decode next to the same GEMV on a freshly allocated matrix of the same
// shape, in one process: decode steps, then the fresh matrix eagerly and from a graph, then
// decode steps again. The JIT disk cache is on so both compiled kernels can be compared.
const STEPS: usize = 12;
const REPS: usize = 10;
const TOKEN: u32 = 760;

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("measurement requires --release".into());
    }
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: lmhead_context_probe MODEL_DIR")?,
    );
    cutile::jit_cache::enable_default()?;
    let device = Device::new(0)?;
    let model = Arc::new(Model::from_checkpoint(
        &path,
        LoadOptions {
            max_seq: 256,
            layers: None,
        },
        &device,
    )?);
    let (rows, cols) = (model.dims().vocab(), model.dims().hidden());
    let mut engine = Engine::new(model, &device)?;
    let mut session = engine.new_session(SessionId(0))?;
    let mut decode = |engine: &mut Engine, label: &str| -> Result<()> {
        println!("PHASE {label} steps={STEPS}");
        for _ in 0..STEPS {
            engine.step(engine.batch([(&mut session, TOKEN)])?)?;
        }
        Ok(())
    };
    decode(&mut engine, "decode-before")?;

    let stream = device.new_stream()?;
    let weight = Weight::Dense(DenseMatrix::upload(
        vec![bf16::from_f32(1.0); rows * cols],
        rows,
        cols,
        &stream,
    )?);
    let tile = GemvTile {
        rows: 16,
        blocks: 4,
    };
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; cols]))
        .reshape(&[1, cols / 32, 32])
        .sync_on(&stream)?;
    let mut y = api::zeros::<f32>(&[1, rows]).sync_on(&stream)?;
    let view = x.view(&[1, cols / 32, 32])?;
    println!("PHASE fresh-eager launches={REPS}");
    for _ in 0..REPS {
        gemv(&weight, &view, (&mut y).partition([1, tile.rows]), tile)
            .submit(&mut Eager(&stream))?;
    }
    let graph = CudaGraph::scope(&stream, |scope| {
        gemv(&weight, &view, (&mut y).partition([1, tile.rows]), tile)
            .submit(&mut Graph(scope))
            .map_err(|e| cuda_async::error::DeviceError::Internal(e.to_string()))
    })?;
    println!("PHASE fresh-graph launches={REPS}");
    for _ in 0..REPS {
        graph.launch().sync_on(&stream)?;
    }
    decode(&mut engine, "decode-after")
}
