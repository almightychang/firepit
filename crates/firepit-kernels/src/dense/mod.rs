mod kernel;

use cuda_async::device_operation::{DeviceOp, GraphNode};
use cutile::DType;
use cutile::api::{self, DeviceOpReshape};
use cutile::core::ElementType;
use cutile::tensor::{Partition, Tensor, TensorView};
use cutile::tile_kernel::TileKernel;
use std::sync::Arc;

use crate::q8::{GemvTile, Q8_BLOCK};

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

pub fn dense_gemv<'a, T: DType + ElementType>(
    w: &DenseMatrix<T>,
    x: &'a TensorView<'a, f32>,
    y: Partition<&'a mut Tensor<f32>>,
    tile: GemvTile,
) -> impl DeviceOp<
    Output = (
        Partition<&'a mut Tensor<f32>>,
        Arc<Tensor<T>>,
        &'a TensorView<'a, f32>,
        i32,
    ),
> + GraphNode
+ 'a {
    let batch = x.shape()[0] as usize;
    assert!(batch > 0 && tile.rows > 0 && tile.blocks > 0);
    assert_eq!(
        x.shape(),
        &[batch as i32, (w.cols / Q8_BLOCK) as i32, Q8_BLOCK as i32]
    );
    assert!(tile.fits(w.rows, w.cols));
    kernel::dense_gemv_kernel(y, w.weights.clone(), x, (w.cols / Q8_BLOCK) as i32).generics(vec![
        T::DTYPE.as_str().to_string(),
        batch.to_string(),
        tile.rows.to_string(),
        tile.blocks.to_string(),
    ])
}
