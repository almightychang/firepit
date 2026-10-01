mod dense;
mod q8;
mod q8_format;

pub use dense::DenseMatrix;
use half::bf16;
pub use q8::Q8Matrix;
pub use q8_format::{FormatError, HostQ8Matrix, Q8_BLOCK};

#[derive(Clone)]
pub enum Weight {
    Dense(DenseMatrix<bf16>),
    Q8(Q8Matrix),
}
