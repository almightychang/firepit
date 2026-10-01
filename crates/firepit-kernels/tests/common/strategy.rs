use proptest::prelude::*;

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

pub fn scalar() -> impl Strategy<Value = f32> {
    prop_oneof![
        5 => prop::collection::vec(-1.0f32..1.0, 6).prop_map(|v| v.into_iter().sum()),
        2 => Just(0.0f32),
        2 => -8.0f32..8.0,
        1 => prop::sample::select(vec![-32.0f32, 32.0]),
    ]
}

pub fn dense_case() -> impl Strategy<Value = DenseCase> {
    (
        1usize..=4,
        1usize..=4,
        prop::sample::select(vec![1usize, 2, 3, 4]),
    )
        .prop_flat_map(|(row_tiles, col_tiles, batch)| {
            let (rows, cols) = (row_tiles * 16, col_tiles * 128);
            (
                Just((rows, cols, batch)),
                prop::collection::vec(scalar(), rows * cols),
                prop::collection::vec(scalar(), batch * cols),
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
        1usize..=4,
        1usize..=4,
        prop::sample::select(vec![1usize, 2, 3, 4]),
    )
        .prop_flat_map(|(row_tiles, col_tiles, batch)| {
            let (rows, cols) = (row_tiles * 16, col_tiles * 128);
            (
                Just((rows, cols, batch)),
                prop::collection::vec(-8i8..9, rows * cols),
                prop::collection::vec(1u8..17, rows * cols / 32),
                prop::collection::vec(scalar(), batch * cols),
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
        prop::sample::select(vec![32usize, 64, 128]),
        prop::sample::select(vec![1usize, 2, 4]),
        prop::sample::select(vec![1e-5f32, 1e-6]),
    )
        .prop_flat_map(|(width, batch, eps)| {
            (
                Just((width, batch, eps)),
                prop::collection::vec(scalar(), width * batch),
                prop::collection::vec(scalar(), width),
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
        prop::sample::select(vec![32usize, 64, 128]),
        prop::sample::select(vec![1usize, 2, 3, 4]),
    )
        .prop_flat_map(|(width, batch)| {
            (
                Just((width, batch)),
                prop::collection::vec(scalar(), 2 * width * batch),
            )
        })
        .prop_map(|((width, batch), gate_up)| SwigluCase {
            width,
            batch,
            gate_up,
        })
}
