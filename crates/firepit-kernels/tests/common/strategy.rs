use proptest::prelude::*;
use proptest::test_runner::TestRng;

#[derive(Clone, Debug)]
pub struct DenseCase {
    pub rows: usize,
    pub cols: usize,
    pub batch: usize,
    pub weights: Vec<f32>,
    pub input: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct Q8Case {
    pub rows: usize,
    pub cols: usize,
    pub batch: usize,
    pub qs: Vec<i8>,
    pub scales: Vec<f32>,
    pub input: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct NormCase {
    pub width: usize,
    pub batch: usize,
    pub input: Vec<f32>,
    pub gamma: Vec<f32>,
    pub eps: f32,
}

#[derive(Clone, Debug)]
pub struct SwigluCase {
    pub width: usize,
    pub batch: usize,
    pub gate_up: Vec<f32>,
}

fn unit(rng: &mut TestRng) -> f32 {
    rng.random_range(-1.0f32..1.0)
}

fn scalar(rng: &mut TestRng) -> f32 {
    match rng.random_range(0..10) {
        0..5 => (0..6).map(|_| unit(rng)).sum(),
        5..7 => 0.0,
        7..9 => 8.0 * unit(rng),
        _ => 32.0f32.copysign(unit(rng)),
    }
}

pub fn values(len: usize) -> impl Strategy<Value = Vec<f32>> {
    Just(len).prop_perturb(|len, mut rng| (0..len).map(|_| scalar(&mut rng)).collect())
}

pub fn dense_case() -> impl Strategy<Value = DenseCase> {
    (
        1usize..=64,
        1usize..=16,
        prop::sample::select(vec![1usize, 2, 3, 4]),
    )
        .prop_flat_map(|(row_tiles, col_tiles, batch)| {
            let (rows, cols) = (row_tiles * 16, col_tiles * 128);
            (
                Just((rows, cols, batch)),
                values(rows * cols),
                values(batch * cols),
            )
        })
        .prop_map(|((rows, cols, batch), weights, input)| DenseCase {
            rows,
            cols,
            batch,
            weights,
            input,
        })
}

pub fn gemm_case() -> impl Strategy<Value = DenseCase> {
    (
        prop::sample::select(vec![128usize, 256]),
        prop::sample::select(vec![128usize, 256, 512]),
        prop::sample::select(vec![1usize, 3, 64, 65, 150]),
    )
        .prop_flat_map(|(rows, cols, batch)| {
            (
                Just((rows, cols, batch)),
                values(rows * cols),
                values(batch * cols),
            )
        })
        .prop_map(|((rows, cols, batch), weights, input)| DenseCase {
            rows,
            cols,
            batch,
            weights,
            input,
        })
}

pub fn q8_case() -> impl Strategy<Value = Q8Case> {
    (
        1usize..=64,
        1usize..=16,
        prop::sample::select(vec![1usize, 2, 3, 4]),
    )
        .prop_flat_map(|(row_tiles, col_tiles, batch)| {
            let (rows, cols) = (row_tiles * 16, col_tiles * 128);
            (
                Just((rows, cols, batch)),
                prop::collection::vec(-8i8..9, rows * cols),
                prop::collection::vec(1u8..17, rows * cols / 32),
                values(batch * cols),
            )
        })
        .prop_map(|((rows, cols, batch), qs, scales, input)| Q8Case {
            rows,
            cols,
            batch,
            qs,
            scales: scales.into_iter().map(|x| x as f32 / 16.0).collect(),
            input,
        })
}

pub fn norm_case() -> impl Strategy<Value = NormCase> {
    (
        prop::sample::select(vec![32usize, 128, 2048]),
        prop::sample::select(vec![1usize, 2, 4]),
        prop::sample::select(vec![1e-5f32, 1e-6]),
    )
        .prop_flat_map(|(width, batch, eps)| {
            (
                Just((width, batch, eps)),
                values(width * batch),
                values(width),
            )
        })
        .prop_map(|((width, batch, eps), input, gamma)| NormCase {
            width,
            batch,
            input,
            gamma,
            eps,
        })
}

pub fn swiglu_case() -> impl Strategy<Value = SwigluCase> {
    (
        prop::sample::select(vec![32usize, 128, 512]),
        prop::sample::select(vec![1usize, 2, 3, 4]),
    )
        .prop_flat_map(|(width, batch)| (Just((width, batch)), values(2 * width * batch)))
        .prop_map(|((width, batch), gate_up)| SwigluCase {
            width,
            batch,
            gate_up,
        })
}
