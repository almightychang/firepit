mod format;
mod kernel;
mod kernel_v2;

use cuda_async::device_operation::DeviceOp;
use cutile::api::{self, DeviceOpReshape};
use cutile::tensor::{Partition, Tensor};
use cutile::tile_kernel::TileKernel;
pub use format::{FormatError, HostQ8Matrix, HostQuantizedX, quantize_x_cpu};
use half::f16;
use std::sync::Arc;

pub struct Q8Matrix {
    pub qs: Arc<Tensor<i8>>,
    pub scales: Arc<Tensor<f16>>,
    pub rows: usize,
    pub cols: usize,
}

impl HostQ8Matrix {
    pub fn upload(
        &self,
        stream: &Arc<cuda_core::Stream>,
    ) -> Result<Q8Matrix, cutile::error::Error> {
        let qs = api::copy_host_vec_to_device(&Arc::new(self.qs.clone()))
            .reshape(&[self.rows, self.cols / 32, 32])
            .sync_on(stream)?;
        let scales = api::copy_host_vec_to_device(&Arc::new(self.scales.clone()))
            .reshape(&[self.rows, self.cols / 32])
            .sync_on(stream)?;
        Ok(Q8Matrix {
            qs: Arc::new(qs),
            scales: Arc::new(scales),
            rows: self.rows,
            cols: self.cols,
        })
    }
}

pub fn q8_gemv<'a, const B: i32, const BN: i32, const NB: i32>(
    w: &Q8Matrix,
    x: &'a Tensor<f32>,
    y: Partition<Tensor<f32>>,
) -> impl DeviceOp<
    Output = (
        Partition<Tensor<f32>>,
        Arc<Tensor<i8>>,
        Arc<Tensor<f16>>,
        &'a Tensor<f32>,
        i32,
    ),
> + 'a {
    assert!(B > 0 && BN > 0 && NB > 0);
    assert_eq!(x.shape(), &[B, (w.cols / 32) as i32, 32]);
    assert_eq!((w.cols / 32) % NB as usize, 0);
    assert_eq!(w.rows % BN as usize, 0);
    kernel::q8_gemv_kernel(y, w.qs.clone(), w.scales.clone(), x, (w.cols / 32) as i32)
        .generics(vec![B.to_string(), BN.to_string(), NB.to_string()])
}

pub fn quantize_x<'a>(
    x: &'a Tensor<f32>,
    xq: Partition<Tensor<i8>>,
    dx: Partition<Tensor<f32>>,
) -> impl DeviceOp<
    Output = (
        Partition<Tensor<i8>>,
        Partition<Tensor<f32>>,
        &'a Tensor<f32>,
    ),
> + 'a {
    assert_eq!(x.shape().len(), 3);
    assert_eq!(x.shape()[2], 32);
    kernel_v2::quantize_x_kernel(xq, dx, x)
}

pub fn q8_gemv_v1b<'a, const B: i32, const BN: i32, const NB: i32>(
    w: &Q8Matrix,
    x: &'a Tensor<f32>,
    y: Partition<Tensor<f32>>,
) -> impl DeviceOp<
    Output = (
        Partition<Tensor<f32>>,
        Arc<Tensor<i8>>,
        Arc<Tensor<f16>>,
        &'a Tensor<f32>,
        i32,
    ),
> + 'a {
    assert!(B > 0 && BN > 0 && NB > 0);
    assert_eq!(x.shape(), &[B, (w.cols / 32) as i32, 32]);
    assert_eq!((w.cols / 32) % NB as usize, 0);
    assert_eq!(w.rows % BN as usize, 0);
    kernel_v2::q8_gemv_v1b_kernel(y, w.qs.clone(), w.scales.clone(), x, (w.cols / 32) as i32)
        .generics(vec![B.to_string(), BN.to_string(), NB.to_string()])
}

pub fn q8_gemv_v2<'a, const B: i32, const BN: i32, const NB: i32>(
    w: &Q8Matrix,
    xq: &'a Tensor<i8>,
    dx: &'a Tensor<f32>,
    y: Partition<Tensor<f32>>,
) -> impl DeviceOp<
    Output = (
        Partition<Tensor<f32>>,
        Arc<Tensor<i8>>,
        Arc<Tensor<f16>>,
        &'a Tensor<i8>,
        &'a Tensor<f32>,
        i32,
    ),
> + 'a {
    assert!(B > 0 && BN > 0 && NB > 0);
    assert_eq!(xq.shape(), &[B, (w.cols / 32) as i32, 32]);
    assert_eq!(dx.shape(), &[B, (w.cols / 32) as i32]);
    assert_eq!((w.cols / 32) % NB as usize, 0);
    assert_eq!(w.rows % BN as usize, 0);
    kernel_v2::q8_gemv_v2_kernel(
        y,
        w.qs.clone(),
        w.scales.clone(),
        xq,
        dx,
        (w.cols / 32) as i32,
    )
    .generics(vec![B.to_string(), BN.to_string(), NB.to_string()])
}
