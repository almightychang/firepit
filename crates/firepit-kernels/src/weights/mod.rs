mod dense;
mod fp8;
mod nvfp4;
mod q8;
mod q8_format;

pub use dense::DenseMatrix;
pub use fp8::{Fp8Matrix, Fp8Scaling, HostFp8Matrix};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixFormat {
    Bf16,
    Fp8(Fp8Scaling),
    Nvfp4,
    Q8,
}

impl Weight {
    pub fn format(&self) -> MatrixFormat {
        match self {
            Self::Dense(_) => MatrixFormat::Bf16,
            Self::Fp8(w) => MatrixFormat::Fp8(w.scaling),
            Self::Nvfp4(_) => MatrixFormat::Nvfp4,
            Self::Q8(_) => MatrixFormat::Q8,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Dense(w) => w.cols,
            Self::Fp8(w) => w.weights.cols,
            Self::Nvfp4(w) => w.cols,
            Self::Q8(w) => w.cols,
        }
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::Dense(w) => w.rows,
            Self::Fp8(w) => w.weights.rows,
            Self::Nvfp4(w) => w.rows,
            Self::Q8(w) => w.rows,
        }
    }
}
