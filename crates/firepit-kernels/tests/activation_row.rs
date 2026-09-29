use cutile::api;
use cutile::prelude::DeviceOp;
use cutile::tensor::IntoPartition;
use cutile::tile_kernel::ToHostVecOp;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    fn set_row(out: &mut Tensor<f32, { [1, 128] }>) {
        let tile: Tile<f32, { [1, 128] }> = constant(3.0f32, shape![1, 128]);
        out.store(tile);
    }
}

#[test]
fn activation_row_isolated_by_mutable_view() {
    let mut activation = api::zeros::<f32>(&[2, 128]).sync().unwrap();
    let b = 1;
    let row = activation.slice_mut(&[b..b + 1, 0..128]).unwrap();
    kernels::set_row(row.partition([1, 128])).sync().unwrap();
    let host: Vec<f32> = activation.dup().to_host_vec().sync().unwrap();
    assert!(host[..128].iter().all(|&x| x == 0.0));
    assert!(host[128..].iter().all(|&x| x == 3.0));
}
