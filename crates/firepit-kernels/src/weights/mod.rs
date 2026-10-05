mod dense;
mod fp8;
mod nvfp4;
mod q8;
mod q8_format;

pub use dense::DenseMatrix;
pub use fp8::{Fp8Matrix, HostFp8Matrix};
use half::bf16;
pub use nvfp4::{HostNvfp4Matrix, NVFP4_GROUP, Nvfp4Matrix};
pub use q8::Q8Matrix;
pub use q8_format::{FormatError, HostQ8Matrix, Q8_BLOCK};

#[derive(Clone)]
pub enum Weight {
    Dense(DenseMatrix<bf16>),
    Fp8(Fp8Matrix),
    Nvfp4(Nvfp4Matrix),
    Q8(Q8Matrix),
}

impl Weight {
    pub fn rows(&self) -> usize {
        match self {
            Self::Dense(w) => w.rows,
            Self::Fp8(w) => w.weights.rows,
            Self::Nvfp4(w) => w.rows,
            Self::Q8(w) => w.rows,
        }
    }
}
