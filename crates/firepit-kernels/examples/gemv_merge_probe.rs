use cuda_async::cuda_graph::CudaGraph;
use cuda_async::device_operation::DeviceOp;
use cuda_async::error::DeviceError;
use cuda_core::{Device, Stream, f8e4m3fn};
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{PartitionMut, Tensor};
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use firepit_kernels::gemv::{DEFAULT_GEMV_TILE, GemvTile, fp8_tile, gemv};
use firepit_kernels::sink::Graph;
use firepit_kernels::weights::{DenseMatrix, HostFp8Matrix, Weight};
use half::bf16;
use std::error::Error;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

// Same-input GEMVs of one decode step, recorded into a CUDA graph either one GEMV per weight
// or with the weights stacked into one matrix, and replayed as decode does. Every layer has
// its own copy of the weights, the process first allocates BALLAST_GIB one-GiB buffers as a
// loaded model would, and each replay follows a read of a 64 MiB buffer. Time with nsys
// (--cuda-graph-trace=node); the GRAPH lines say how many kernels each replay launches.
const WARMUP: usize = 1;
const REPLAYS: usize = 10;
const SCRUB_TILE: usize = 4096;
const SCRUB_LEN: usize = 16 << 20;
const HIDDEN: usize = 2048;
const KT: usize = 128;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn scrub_read<const B: i32>(
        sums: &mut Tensor<f32, { [1] }>,
        buffer: &Tensor<f32, { [-1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let values: Tile<f32, { [B] }> = buffer.partition(shape![B]).load([pid.0]);
        let sum: Tile<f32, { [] }> = reduce_sum(values, 0i32);
        sums.store(sum.reshape(shape![1]));
    }

    // dense_fp8_kernel's body with one K loop and a row-scale vector applied at the end.
    #[cutile::entry()]
    pub fn fp8_flat<const B: i32, const BN: i32, const NB: i32, const KT: i32>(
        y: &mut Tensor<f32, { [B, BN] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        row_scales: &Tensor<f32, { [1, -1] }>,
        x: &Tensor<f32, { [B, -1, 32] }>,
        chunks: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let w_part = w.partition(shape![BN, KT]);
        let x_part = x.partition(shape![B, NB, 32]);
        let mut acc: Tile<f32, { [B, BN] }> = constant(0.0f32, shape![B, BN]);
        for kc in 0i32..chunks {
            let weights: Tile<f8e4m3fn, { [BN, KT] }> = w_part.load([pid.1, kc]);
            let weights: Tile<f32, { [BN, KT] }> = convert_tile(weights);
            let weights: Tile<f32, { [B, BN, KT] }> = weights
                .reshape(shape![1, BN, KT])
                .broadcast(shape![B, BN, KT]);
            let xb: Tile<f32, { [B, NB, 32] }> = x_part.load([0, kc, 0]);
            let xb: Tile<f32, { [B, BN, KT] }> =
                xb.reshape(shape![B, 1, KT]).broadcast(shape![B, BN, KT]);
            let part: Tile<f32, { [B, BN] }> = reduce_sum(weights * xb, 2i32);
            acc = acc + part.reshape(shape![B, BN]);
        }
        let row_scale: Tile<f32, { [1, BN] }> =
            row_scales.partition(shape![1, BN]).load([0, pid.1]);
        y.store(acc * row_scale.broadcast(shape![B, BN]));
    }
}

struct Lcg(u64);

impl Lcg {
    fn byte(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 56) as u8
    }
}

// An FP8 matrix with one scale per stacked part, kept both in the product per-tensor form
// (one Weight per part) and as raw bytes with a row-scale vector for the flat kernel.
struct Fp8Parts {
    rows: Vec<usize>,
    bytes: Vec<Vec<f8e4m3fn>>,
    scales: Vec<f32>,
}

impl Fp8Parts {
    fn new(rows: &[usize], rng: &mut Lcg) -> Self {
        Self {
            rows: rows.to_vec(),
            bytes: rows
                .iter()
                .map(|&r| {
                    (0..r * HIDDEN)
                        .map(|_| f8e4m3fn(rng.byte() & 0x3f | 0x20))
                        .collect()
                })
                .collect(),
            scales: rows
                .iter()
                .enumerate()
                .map(|(i, _)| 0.25 * (i + 1) as f32)
                .collect(),
        }
    }

    fn per_tensor(&self, stream: &Arc<Stream>) -> Result<Vec<(Weight, GemvTile)>> {
        self.rows
            .iter()
            .zip(&self.bytes)
            .zip(&self.scales)
            .map(|((&rows, bytes), &scale)| {
                let weight = Weight::Fp8(
                    HostFp8Matrix::from_parts(
                        bytes.clone(),
                        vec![scale],
                        rows,
                        HIDDEN,
                        rows,
                        HIDDEN,
                    )?
                    .upload(stream)?,
                );
                Ok((weight, fp8_tile(rows, HIDDEN, rows, HIDDEN, 1)))
            })
            .collect()
    }

    fn flat(&self, stream: &Arc<Stream>, merged: bool) -> Result<Vec<FlatMatrix>> {
        let parts: Vec<(usize, Vec<f8e4m3fn>, Vec<f32>)> = if merged {
            vec![(
                self.rows.iter().sum(),
                self.bytes.concat(),
                self.rows
                    .iter()
                    .zip(&self.scales)
                    .flat_map(|(&r, &s)| std::iter::repeat_n(s, r))
                    .collect(),
            )]
        } else {
            self.rows
                .iter()
                .zip(&self.bytes)
                .zip(&self.scales)
                .map(|((&r, b), &s)| (r, b.clone(), vec![s; r]))
                .collect()
        };
        parts
            .into_iter()
            .map(|(rows, bytes, scales)| {
                Ok(FlatMatrix {
                    rows,
                    weights: Arc::new(
                        api::copy_host_vec_to_device(&Arc::new(bytes))
                            .reshape(&[rows, HIDDEN])
                            .sync_on(stream)?,
                    ),
                    row_scales: Arc::new(
                        api::copy_host_vec_to_device(&Arc::new(scales))
                            .reshape(&[1, rows])
                            .sync_on(stream)?,
                    ),
                    tile: fp8_tile(rows, HIDDEN, rows, HIDDEN, 1),
                })
            })
            .collect()
    }
}

struct FlatMatrix {
    rows: usize,
    weights: Arc<Tensor<f8e4m3fn>>,
    row_scales: Arc<Tensor<f32>>,
    tile: GemvTile,
}

fn dense_tile(rows: usize) -> GemvTile {
    let mut tile = DEFAULT_GEMV_TILE;
    while rows % tile.rows != 0 {
        tile.rows /= 2;
    }
    while (HIDDEN / 32) % tile.blocks != 0 {
        tile.blocks /= 2;
    }
    tile
}

fn bf16_parts(rows: &[usize], rng: &mut Lcg) -> Vec<Vec<bf16>> {
    rows.iter()
        .map(|&r| {
            (0..r * HIDDEN)
                .map(|_| bf16::from_f32((rng.byte() as f32 - 128.0) / 256.0))
                .collect()
        })
        .collect()
}

fn dense(stream: &Arc<Stream>, rows: usize, data: Vec<bf16>) -> Result<(Weight, GemvTile)> {
    Ok((
        Weight::Dense(DenseMatrix::upload(data, rows, HIDDEN, stream)?),
        dense_tile(rows),
    ))
}

// One layer's GEMVs in one form: product GEMVs (per-tensor FP8 or bf16) and flat FP8 matrices.
struct Layer {
    product: Vec<(Weight, GemvTile)>,
    flat: Vec<FlatMatrix>,
    outputs: Vec<Tensor<f32>>,
}

impl Layer {
    fn new(
        stream: &Arc<Stream>,
        product: Vec<(Weight, GemvTile)>,
        flat: Vec<FlatMatrix>,
    ) -> Result<Self> {
        let mut outputs = Vec::new();
        for (weight, _) in &product {
            let rows = match weight {
                Weight::Fp8(m) => m.weights.rows,
                Weight::Dense(m) => m.rows,
                _ => unreachable!(),
            };
            outputs.push(api::zeros::<f32>(&[1, rows]).sync_on(stream)?);
        }
        for matrix in &flat {
            outputs.push(api::zeros::<f32>(&[1, matrix.rows]).sync_on(stream)?);
        }
        Ok(Self {
            product,
            flat,
            outputs,
        })
    }

    fn kernels(&self) -> usize {
        self.product.len() + self.flat.len()
    }
}

struct Probe {
    stream: Arc<Stream>,
    buffer: Arc<Tensor<f32>>,
    sums: Tensor<f32>,
    x: Arc<Tensor<f32>>,
}

impl Probe {
    fn run(&mut self, name: &str, layers: &mut [Layer]) -> Result<Vec<Vec<f32>>> {
        let x = self.x.clone();
        let view = x.view(&[1, HIDDEN / 32, 32])?;
        let graph = CudaGraph::scope(&self.stream, |scope| {
            for layer in layers.iter_mut() {
                let (product_outputs, flat_outputs) =
                    layer.outputs.split_at_mut(layer.product.len());
                for ((weight, tile), y) in layer.product.iter().zip(product_outputs.iter_mut()) {
                    gemv(weight, &view, y.partition([1, tile.rows]), *tile)
                        .submit(&mut Graph(scope))
                        .map_err(|e| DeviceError::Internal(e.to_string()))?;
                }
                for (matrix, y) in layer.flat.iter().zip(flat_outputs.iter_mut()) {
                    scope.record(
                        kernels::fp8_flat(
                            y.partition([1, matrix.tile.rows]),
                            matrix.weights.clone(),
                            matrix.row_scales.clone(),
                            &view,
                            (HIDDEN / KT) as i32,
                        )
                        .generics(vec![
                            "1".into(),
                            matrix.tile.rows.to_string(),
                            (KT / 32).to_string(),
                            KT.to_string(),
                        ]),
                    )?;
                }
            }
            Ok(())
        })?;
        let per_replay: usize = layers.iter().map(Layer::kernels).sum();
        println!(
            "GRAPH {name} kernels={per_replay} replays={}",
            WARMUP + REPLAYS
        );
        for _ in 0..WARMUP + REPLAYS {
            kernels::scrub_read((&mut self.sums).partition([1]), self.buffer.clone())
                .generics(vec![SCRUB_TILE.to_string()])
                .sync_on(&self.stream)?;
            graph.launch().sync_on(&self.stream)?;
        }
        layers[0]
            .outputs
            .iter()
            .map(|y| Ok(y.dup().to_host_vec().sync_on(&self.stream)?))
            .collect()
    }
}

fn max_relative(a: &[f32], b: &[f32]) -> f32 {
    let scale = b
        .iter()
        .fold(0.0f32, |m, v| m.max(v.abs()))
        .max(f32::MIN_POSITIVE);
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (p, q)| m.max((p - q).abs()))
        / scale
}

fn report(name: &str, separate: &[Vec<f32>], merged: &[Vec<f32>]) {
    let joined: Vec<f32> = separate.concat();
    println!(
        "CHECK {name} max_rel {:.3e}",
        max_relative(&merged.concat(), &joined)
    );
}

fn main() -> Result<()> {
    if cfg!(debug_assertions) {
        return Err("measurement requires --release".into());
    }
    let ballast_gib: usize = std::env::args()
        .nth(1)
        .ok_or("usage: gemv_merge_probe BALLAST_GIB")?
        .parse()?;
    cutile::jit_cache::enable_default()?;
    let stream: Arc<Stream> = Device::new(0)?.new_stream()?;
    let ballast: Vec<Tensor<f32>> = (0..ballast_gib)
        .map(|_| api::zeros::<f32>(&[1 << 28]).sync_on(&stream))
        .collect::<std::result::Result<_, _>>()?;
    let mut rng = Lcg(11);
    let mut probe = Probe {
        buffer: Arc::new(api::zeros::<f32>(&[SCRUB_LEN]).sync_on(&stream)?),
        sums: api::zeros::<f32>(&[SCRUB_LEN / SCRUB_TILE]).sync_on(&stream)?,
        x: Arc::new(
            api::copy_host_vec_to_device(&Arc::new(
                (0..HIDDEN)
                    .map(|i| ((i * 7 % 13) as f32 - 6.0) / 6.0)
                    .collect::<Vec<f32>>(),
            ))
            .reshape(&[1, HIDDEN / 32, 32])
            .sync_on(&stream)?,
        ),
        stream: stream.clone(),
    };

    // GDN, 30 layers: FP8 qkv 8192 + z 4096 rows, bf16 b 32 + a 32 rows.
    let fp8 = Fp8Parts::new(&[8192, 4096], &mut rng);
    let ba = bf16_parts(&[32, 32], &mut rng);
    let build = |form: &str| -> Result<Vec<Layer>> {
        (0..30)
            .map(|_| {
                let bf16_product = |merged: bool| -> Result<Vec<(Weight, GemvTile)>> {
                    if merged {
                        Ok(vec![dense(&stream, 64, ba.concat())?])
                    } else {
                        ba.iter().map(|d| dense(&stream, 32, d.clone())).collect()
                    }
                };
                match form {
                    "separate" => Layer::new(
                        &stream,
                        [fp8.per_tensor(&stream)?, bf16_product(false)?].concat(),
                        vec![],
                    ),
                    "separate-flat" => {
                        Layer::new(&stream, bf16_product(false)?, fp8.flat(&stream, false)?)
                    }
                    _ => Layer::new(&stream, bf16_product(true)?, fp8.flat(&stream, true)?),
                }
            })
            .collect()
    };
    let separate = probe.run("gdn-separate", &mut build("separate")?)?;
    let separate_flat = probe.run("gdn-separate-flat", &mut build("separate-flat")?)?;
    let merged = probe.run("gdn-merged", &mut build("merged")?)?;
    report(
        "gdn separate-flat vs separate (fp8 parts)",
        &separate[..2],
        &[separate_flat[2].clone(), separate_flat[3].clone()],
    );
    report(
        "gdn merged vs separate (fp8)",
        &separate[..2],
        &merged[1..2],
    );
    report(
        "gdn merged vs separate (bf16)",
        &separate[2..4],
        &merged[..1],
    );

    // Attention, 10 layers: FP8 q_gate 8192 + k 512 + v 512 rows.
    let fp8 = Fp8Parts::new(&[8192, 512, 512], &mut rng);
    let build = |form: &str| -> Result<Vec<Layer>> {
        (0..10)
            .map(|_| match form {
                "separate" => Layer::new(&stream, fp8.per_tensor(&stream)?, vec![]),
                "separate-flat" => Layer::new(&stream, vec![], fp8.flat(&stream, false)?),
                _ => Layer::new(&stream, vec![], fp8.flat(&stream, true)?),
            })
            .collect()
    };
    let separate = probe.run("attention-separate", &mut build("separate")?)?;
    let _ = probe.run("attention-separate-flat", &mut build("separate-flat")?)?;
    let merged = probe.run("attention-merged", &mut build("merged")?)?;
    report("attention merged vs separate", &separate, &merged);

    // Router and shared expert gate, 40 layers: bf16 256 + 1 rows, stacked as 257 rows or
    // padded with zero rows to 264.
    let parts = bf16_parts(&[256, 1], &mut rng);
    let build = |rows: usize| -> Result<Vec<Layer>> {
        (0..40)
            .map(|_| {
                if rows == 0 {
                    Layer::new(
                        &stream,
                        vec![
                            dense(&stream, 256, parts[0].clone())?,
                            dense(&stream, 1, parts[1].clone())?,
                        ],
                        vec![],
                    )
                } else {
                    let mut data = parts.concat();
                    data.resize(rows * HIDDEN, bf16::from_f32(0.0));
                    Layer::new(&stream, vec![dense(&stream, rows, data)?], vec![])
                }
            })
            .collect()
    };
    let separate = probe.run("router-separate", &mut build(0)?)?;
    let merged = probe.run("router-merged-257", &mut build(257)?)?;
    let padded = probe.run("router-merged-264", &mut build(264)?)?;
    report("router merged-257 vs separate", &separate, &merged);
    report(
        "router merged-264 vs separate",
        &separate,
        &[padded[0][..257].to_vec()],
    );
    drop(ballast);
    Ok(())
}
