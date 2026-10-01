use super::{HostQ8Matrix, Q8_BLOCK};
use cuda_async::device_operation::DeviceOp;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::Tensor;
use half::f16;
use std::sync::Arc;

#[derive(Clone)]
pub struct Q8Matrix {
    pub(crate) qs: Arc<Tensor<i8>>,
    pub(crate) scales: Arc<Tensor<f16>>,
    pub rows: usize,
    pub(crate) cols: usize,
}

impl HostQ8Matrix {
    pub fn upload(
        &self,
        stream: &Arc<cuda_core::Stream>,
    ) -> Result<Q8Matrix, cutile::error::Error> {
        let qs = api::copy_host_vec_to_device(&Arc::new(self.qs.clone()))
            .reshape(&[self.rows, self.cols / Q8_BLOCK, Q8_BLOCK])
            .sync_on(stream)?;
        let scales = api::copy_host_vec_to_device(&Arc::new(self.scales.clone()))
            .reshape(&[self.rows, self.cols / Q8_BLOCK])
            .sync_on(stream)?;
        Ok(Q8Matrix {
            qs: Arc::new(qs),
            scales: Arc::new(scales),
            rows: self.rows,
            cols: self.cols,
        })
    }
}
