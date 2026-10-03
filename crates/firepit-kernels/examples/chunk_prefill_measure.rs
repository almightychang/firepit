use cuda_async::device_operation::DeviceOp;
use cuda_core::{Device, Stream};
use cutile::api::{self, DeviceOpReshape};
use cutile::bench::{BenchOptions, do_bench};
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::TileKernel;
use firepit_kernels::gdn_chunk::{
    apply_inverse, first_phase_matrices, invert_diagonal, merge_inverse, second_phase,
    second_phase_value_tile,
};
use std::error::Error;
use std::sync::Arc;

const CHUNK: usize = 64;
const BLOCK: usize = 16;
const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 32;
const DK: usize = 128;
const DV: usize = 128;

struct Buffers {
    chunks: usize,
    q: Tensor<f32>,
    k: Tensor<f32>,
    v: Tensor<f32>,
    g: Tensor<f32>,
    beta: Tensor<f32>,
    ut: Tensor<f32>,
    intra: Tensor<f32>,
    cum: Tensor<f32>,
    diagonals: Tensor<f32>,
    inverse: Tensor<f32>,
    next_inverse: Tensor<f32>,
    u: Tensor<f32>,
    w: Tensor<f32>,
    core: Tensor<f32>,
    state: Tensor<f32>,
    value_tile: usize,
}

impl Buffers {
    fn new(stream: &Arc<Stream>, time: usize) -> Result<Self, Box<dyn Error>> {
        let chunks = time.div_ceil(CHUNK);
        let flat = chunks * VALUE_HEADS;
        let device = Device::new(0)?;
        let sm_count =
            cuda_core::simt::CudaContext::new(device.ordinal())?.multiprocessor_count()?;
        let value_tile = second_phase_value_tile(DV, VALUE_HEADS, sm_count as usize);
        let upload = |data: Vec<f32>, shape: &[usize]| {
            api::copy_host_vec_to_device(&Arc::new(data))
                .reshape(shape)
                .sync_on(stream)
        };
        let qk_len = chunks * CHUNK * KEY_HEADS * DK;
        let v_len = chunks * CHUNK * VALUE_HEADS * DV;
        let gate_len = chunks * CHUNK * VALUE_HEADS;
        let padded = |len: usize, valid: usize, value: f32| {
            let mut data = vec![0.0; len];
            data[..valid].fill(value);
            data
        };
        let q = upload(
            padded(qk_len, time * KEY_HEADS * DK, 0.25),
            &[chunks, CHUNK, KEY_HEADS, DK],
        )?;
        let k = upload(
            padded(qk_len, time * KEY_HEADS * DK, 0.125),
            &[chunks, CHUNK, KEY_HEADS, DK],
        )?;
        let v = upload(
            padded(v_len, time * VALUE_HEADS * DV, 0.5),
            &[chunks, CHUNK, VALUE_HEADS, DV],
        )?;
        let g = upload(
            padded(gate_len, time * VALUE_HEADS, -0.03),
            &[chunks, CHUNK, VALUE_HEADS],
        )?;
        let beta = upload(
            padded(gate_len, time * VALUE_HEADS, 0.4),
            &[chunks, CHUNK, VALUE_HEADS],
        )?;
        Ok(Self {
            chunks,
            q,
            k,
            v,
            g,
            beta,
            ut: api::zeros::<f32>(&[flat, CHUNK, CHUNK]).sync_on(stream)?,
            intra: api::zeros::<f32>(&[flat, CHUNK, CHUNK]).sync_on(stream)?,
            cum: api::zeros::<f32>(&[flat, CHUNK]).sync_on(stream)?,
            diagonals: api::zeros::<f32>(&[flat * CHUNK / BLOCK, BLOCK, BLOCK]).sync_on(stream)?,
            inverse: api::zeros::<f32>(&[flat, CHUNK, CHUNK]).sync_on(stream)?,
            next_inverse: api::zeros::<f32>(&[flat, CHUNK, CHUNK]).sync_on(stream)?,
            u: api::zeros::<f32>(&[flat, CHUNK, DV]).sync_on(stream)?,
            w: api::zeros::<f32>(&[flat, CHUNK, DK]).sync_on(stream)?,
            core: api::zeros::<f32>(&[chunks * CHUNK, VALUE_HEADS, DV]).sync_on(stream)?,
            state: api::zeros::<f32>(&[1, VALUE_HEADS * DK, DV]).sync_on(stream)?,
            value_tile,
        })
    }

    fn first(&mut self, stream: &Arc<Stream>) -> Result<(), cutile::error::Error> {
        first_phase_matrices(
            (&mut self.ut).partition([1, CHUNK, CHUNK]),
            (&mut self.intra).partition([1, CHUNK, CHUNK]),
            (&mut self.cum).partition([1, CHUNK]),
            &self.q,
            &self.k,
            &self.g,
            &self.beta,
            CHUNK as i32,
            1e-6,
            1.0 / (DK as f32).sqrt(),
        )
        .generics(vec![
            CHUNK.to_string(),
            DK.to_string(),
            KEY_HEADS.to_string(),
            VALUE_HEADS.to_string(),
        ])
        .sync_on(stream)
        .map(|_| ())
        .map_err(Into::into)
    }

    fn diagonal(&mut self, stream: &Arc<Stream>) -> Result<(), cutile::error::Error> {
        invert_diagonal((&mut self.diagonals).partition([1, BLOCK, BLOCK]), &self.ut)
            .generics(vec![CHUNK.to_string(), BLOCK.to_string()])
            .sync_on(stream)
            .map(|_| ())
            .map_err(Into::into)
    }

    fn merge(&mut self, stream: &Arc<Stream>) -> Result<(), cutile::error::Error> {
        for row in 0..CHUNK / BLOCK {
            merge_inverse(
                (&mut self.next_inverse).partition([1, CHUNK, CHUNK]),
                &self.inverse,
                &self.ut,
                &self.diagonals,
                row as i32,
            )
            .generics(vec![CHUNK.to_string(), BLOCK.to_string()])
            .sync_on(stream)
            .map_err(cutile::error::Error::from)?;
            std::mem::swap(&mut self.inverse, &mut self.next_inverse);
        }
        Ok(())
    }

    fn apply(&mut self, stream: &Arc<Stream>) -> Result<(), cutile::error::Error> {
        apply_inverse(
            (&mut self.u).partition([1, CHUNK, DV]),
            (&mut self.w).partition([1, CHUNK, DK]),
            &self.inverse,
            &self.cum,
            &self.k,
            &self.v,
            &self.beta,
            CHUNK as i32,
            1e-6,
        )
        .generics(vec![
            CHUNK.to_string(),
            DK.to_string(),
            DV.to_string(),
            KEY_HEADS.to_string(),
            VALUE_HEADS.to_string(),
        ])
        .sync_on(stream)
        .map(|_| ())
        .map_err(Into::into)
    }

    fn all(&mut self, stream: &Arc<Stream>) -> Result<(), cutile::error::Error> {
        self.first(stream)?;
        self.diagonal(stream)?;
        self.merge(stream)?;
        self.apply(stream)?;
        self.second(stream)
    }

    fn second(&mut self, stream: &Arc<Stream>) -> Result<(), cutile::error::Error> {
        second_phase(
            (&mut self.core).partition([
                (self.chunks * CHUNK).next_power_of_two(),
                1,
                self.value_tile,
            ]),
            (&mut self.state).partition([1, DK, self.value_tile]),
            &self.u,
            &self.w,
            &self.intra,
            &self.cum,
            &self.q,
            &self.k,
            self.chunks as i32,
            1e-6,
            1.0 / (DK as f32).sqrt(),
        )
        .generics(vec![
            (self.chunks * CHUNK).next_power_of_two().to_string(),
            CHUNK.to_string(),
            DK.to_string(),
            self.value_tile.to_string(),
            KEY_HEADS.to_string(),
            VALUE_HEADS.to_string(),
        ])
        .sync_on(stream)
        .map(|_| ())
        .map_err(Into::into)
    }
}

fn measure(stream: &Arc<Stream>, time: usize) -> Result<(), Box<dyn Error>> {
    if time == 0 || time > i32::MAX as usize / (VALUE_HEADS * DK) {
        return Err("invalid T".into());
    }
    let mut buffers = Buffers::new(stream, time)?;
    buffers.all(stream)?;
    let options = BenchOptions {
        clear_l2: false,
        ..BenchOptions::default()
    };
    let first = do_bench(stream, &options, |stream| buffers.first(stream))?;
    let diagonal = do_bench(stream, &options, |stream| buffers.diagonal(stream))?;
    let merge = do_bench(stream, &options, |stream| buffers.merge(stream))?;
    let apply = do_bench(stream, &options, |stream| buffers.apply(stream))?;
    let second = do_bench(stream, &options, |stream| buffers.second(stream))?;
    let total = do_bench(stream, &options, |stream| buffers.all(stream))?;
    println!(
        "T={time} chunks={} Hv={VALUE_HEADS} Hk={KEY_HEADS} dk={DK} dv={DV} chunk={CHUNK} block={BLOCK} value_tile={} first_ms={:.6} diagonal_ms={:.6} merge_ms={:.6} apply_ms={:.6} second_ms={:.6} total_ms={:.6} reps={}/{}/{}/{}/{}/{}",
        buffers.chunks,
        buffers.value_tile,
        first.median_ms(),
        diagonal.median_ms(),
        merge.median_ms(),
        apply.median_ms(),
        second.median_ms(),
        total.median_ms(),
        first.reps(),
        diagonal.reps(),
        merge.reps(),
        apply.reps(),
        second.reps(),
        total.reps(),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    assert!(!cfg!(debug_assertions), "measurement requires --release");
    cutile::jit_cache::enable_default()?;
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    println!("device={} clear_l2=false", device.name()?);
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|arg| arg.parse())
        .collect::<Result<_, _>>()?;
    let times = if args.is_empty() {
        vec![512, 2048, 5000]
    } else {
        args
    };
    for time in times {
        measure(&stream, time)?;
    }
    Ok(())
}
