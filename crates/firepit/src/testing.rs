use crate::checkpoint::Stream;
use crate::model::{AttentionWeights, GdnWeights, LayerWeights, MixerWeights, Model, MoeWeights};
use crate::{LayerKind, ModelConfig, Result};
use cuda_async::device_operation::DeviceOp;
use cutile::tensor::Tensor;
use firepit_kernels::weights::{DenseMatrix, Weight};
use half::bf16;
use serde_json::Value;
use std::sync::Arc;

impl Model {
    pub fn random_from_config(
        text: &Value,
        max_seq: usize,
        seed: u64,
        stream: &Stream,
    ) -> Result<Self> {
        let dims = ModelConfig::try_from(text.clone())?.dims(max_seq, None)?;
        let h = dims.hidden();
        let mut random = RandomWeights(seed);
        let embedding = random.tensor(&[dims.vocab(), h], stream)?;
        let mut layers = Vec::with_capacity(dims.layers().len());
        for &kind in dims.layers() {
            let input_norm = random.norm(h, stream)?;
            let mixer = match kind {
                LayerKind::Gdn => {
                    let g = dims.gdn();
                    MixerWeights::Gdn(GdnWeights {
                        qkv: random.matrix(g.conv_dim(), h, stream)?,
                        z: random.matrix(g.value_dim(), h, stream)?,
                        b: random.matrix(g.value_heads(), h, stream)?,
                        a: random.matrix(g.value_heads(), h, stream)?,
                        conv: random.tensor(&[g.conv_dim(), g.conv_kernel()], stream)?,
                        a_log: random.tensor(&[g.value_heads()], stream)?,
                        dt_bias: random.tensor(&[g.value_heads()], stream)?,
                        gated_norm: random.norm(g.value_head_dim(), stream)?,
                        out: random.matrix(h, g.value_dim(), stream)?,
                    })
                }
                LayerKind::Attention => {
                    let a = dims.attention();
                    MixerWeights::Attention(AttentionWeights {
                        q_gate: random.matrix(2 * a.q_heads() * a.head_dim(), h, stream)?,
                        k: random.matrix(a.kv_heads() * a.head_dim(), h, stream)?,
                        v: random.matrix(a.kv_heads() * a.head_dim(), h, stream)?,
                        q_norm: random.norm(a.head_dim(), stream)?,
                        k_norm: random.norm(a.head_dim(), stream)?,
                        out: random.matrix(h, a.q_heads() * a.head_dim(), stream)?,
                    })
                }
            };
            let post_norm = random.norm(h, stream)?;
            let m = dims.moe();
            let stacked = m.with_shared_expert().experts();
            let gate_up = random.dense(stacked * 2 * m.intermediate(), h, stream)?;
            let down = random.dense(stacked * h, m.intermediate(), stream)?;
            let moe = MoeWeights {
                router: random.matrix(m.experts(), h, stream)?,
                gate_up: Weight::Dense(gate_up),
                down: Weight::Dense(down),
                shared_gate: random.matrix(1, h, stream)?,
            };
            layers.push(LayerWeights {
                input_norm,
                mixer,
                post_norm,
                moe,
            });
        }
        let final_norm = random.norm(h, stream)?;
        let lm_head = random.matrix(dims.vocab(), h, stream)?;
        Ok(Self {
            device_ordinal: stream.device().ordinal(),
            dims,
            embedding,
            layers,
            final_norm,
            lm_head,
        })
    }
}

struct RandomWeights(u64);

impl RandomWeights {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as u32 as f32 / u32::MAX as f32) * 0.4 - 0.2
    }

    fn tensor(&mut self, shape: &[usize], stream: &Stream) -> Result<Tensor<f32>> {
        use cutile::api::{self, DeviceOpReshape};
        let data = (0..shape.iter().product()).map(|_| self.next()).collect();
        Ok(api::copy_host_vec_to_device(&Arc::new(data))
            .reshape(shape)
            .sync_on(stream)?)
    }

    fn norm(&mut self, len: usize, stream: &Stream) -> Result<Tensor<f32>> {
        use cutile::api;
        let data = (0..len).map(|_| 1.0 + self.next()).collect();
        Ok(api::copy_host_vec_to_device(&Arc::new(data)).sync_on(stream)?)
    }

    fn dense(&mut self, rows: usize, cols: usize, stream: &Stream) -> Result<DenseMatrix<bf16>> {
        let data = (0..rows * cols)
            .map(|_| bf16::from_f32(self.next()))
            .collect();
        Ok(DenseMatrix::upload(data, rows, cols, stream)?)
    }

    fn matrix(&mut self, rows: usize, cols: usize, stream: &Stream) -> Result<Weight> {
        Ok(Weight::Dense(self.dense(rows, cols, stream)?))
    }
}
