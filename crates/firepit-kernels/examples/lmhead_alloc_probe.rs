use cuda_async::device_operation::DeviceOp;
use cuda_core::Device;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, Tensor};
use firepit_kernels::gemv::{GemvTile, gemv};
use firepit_kernels::sink::Eager;
use firepit_kernels::weights::{DenseMatrix, Weight};
use half::bf16;
use std::error::Error;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// The lm_head GEMV on a fresh matrix after first allocating and filling BALLAST_GIB one-GiB
// device buffers (0 for none), to see whether earlier large allocations change its time.
const REPS: usize = 10;
const ROWS: usize = 248320;
const COLS: usize = 2048;

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("measurement requires --release".into());
    }
    let ballast_gib: usize = std::env::args()
        .nth(1)
        .ok_or("usage: lmhead_alloc_probe BALLAST_GIB")?
        .parse()?;
    let stream = Device::new(0)?.new_stream()?;
    let ballast: Vec<Tensor<f32>> = (0..ballast_gib)
        .map(|_| api::zeros::<f32>(&[1 << 28]).sync_on(&stream))
        .collect::<std::result::Result<_, _>>()?;
    println!("BALLAST {} GiB in {} buffers", ballast_gib, ballast.len());
    let weight = Weight::Dense(DenseMatrix::upload(
        vec![bf16::from_f32(1.0); ROWS * COLS],
        ROWS,
        COLS,
        &stream,
    )?);
    let tile = GemvTile {
        rows: 16,
        blocks: 4,
    };
    let x = api::copy_host_vec_to_device(&Arc::new(vec![1.0f32; COLS]))
        .reshape(&[1, COLS / 32, 32])
        .sync_on(&stream)?;
    let mut y = api::zeros::<f32>(&[1, ROWS]).sync_on(&stream)?;
    println!("PHASE lmhead ballast={ballast_gib} launches={REPS}");
    for _ in 0..REPS {
        gemv(
            &weight,
            &x.view(&[1, COLS / 32, 32])?,
            (&mut y).partition([1, tile.rows]),
            tile,
        )
        .submit(&mut Eager(&stream))?;
    }
    drop(ballast);
    Ok(())
}
