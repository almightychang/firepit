use cuda_async::device_operation::DeviceOp;
use cutile::DType;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::Tensor;
use std::sync::Arc;

use super::Q8_BLOCK;

#[derive(Clone)]
pub struct DenseMatrix<T: DType> {
    pub(crate) weights: Arc<Tensor<T>>,
    pub rows: usize,
    pub(crate) cols: usize,
}

impl<T: DType> DenseMatrix<T> {
    pub fn upload(
        data: Vec<T>,
        rows: usize,
        cols: usize,
        stream: &Arc<cuda_core::Stream>,
    ) -> Result<Self, cutile::error::Error> {
        assert!(rows > 0 && cols > 0 && cols % Q8_BLOCK == 0);
        assert_eq!(data.len(), rows * cols);
        let weights = api::copy_host_vec_to_device(&Arc::new(data))
            .reshape(&[rows, cols / Q8_BLOCK, Q8_BLOCK])
            .sync_on(stream)?;
        Ok(Self {
            weights: Arc::new(weights),
            rows,
            cols,
        })
    }
}
