use cutile::api;
use cutile::compile_api::KernelCompiler;
use cutile::cutile_compiler::cuda_tile_runtime_utils::get_gpu_name;
use cutile::cutile_compiler::specialization::{SpecializationBits, compute_spec};
use cutile::prelude::DeviceOp;
use cutile::tensor::PartitionMut;
use cutile::tile_kernel::{TileKernel, ToHostVecOp};
use std::error::Error;
use std::sync::Arc;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn write_row_static<const ROWS: i32, const D: i32>(
        slab: &mut Tensor<f32, { [ROWS, D] }>,
        row: &Tensor<f32, { [1, D] }>,
        index: i32,
    ) {
        let tile: Tile<f32, { [1, D] }> = row.partition(shape![1, D]).load([0, 0]);
        slab.partition_mut(shape![1, D]).store(tile, [index, 0]);
    }

    #[cutile::entry()]
    pub fn write_row_dynamic<const D: i32>(
        slab: &mut Tensor<f32, { [-1, D] }>,
        row: &Tensor<f32, { [1, D] }>,
        index: i32,
    ) {
        let tile: Tile<f32, { [1, D] }> = row.partition(shape![1, D]).load([0, 0]);
        slab.partition_mut(shape![1, D]).store(tile, [index, 0]);
    }
}

const ROWS: usize = 8;
const D: usize = 32;
const INDEX: i32 = 3;

fn contiguous(shape: &[i32]) -> Vec<i32> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len() - 1).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

fn compile_dynamic_ir() -> Result<String, Box<dyn Error>> {
    let slab = [ROWS as i32, D as i32];
    let row = [1, D as i32];
    let arguments: [(&str, &[i32]); 2] = [("slab", &slab), ("row", &row)];
    let hints: Vec<(&str, Vec<i32>)> = arguments
        .iter()
        .map(|&(name, shape)| {
            let hint = contiguous(shape)
                .iter()
                .map(|&stride| if stride == 1 { 1 } else { -1 })
                .collect();
            (name, hint)
        })
        .collect();
    let strides: Vec<(&str, &[i32])> = hints.iter().map(|(n, h)| (*n, h.as_slice())).collect();
    let specs: Vec<(&str, SpecializationBits)> = arguments
        .iter()
        .map(|&(name, shape)| (name, compute_spec(16, shape, &contiguous(shape), 4)))
        .collect();
    Ok(
        KernelCompiler::new(kernels::__module_ast_self, "kernels", "write_row_dynamic")
            .target(&get_gpu_name(0))
            .generics(vec![D.to_string()])
            .strides(&strides)
            .spec_args(&specs)
            .compile()?
            .ir_text(),
    )
}

fn written_row(slab: &[f32]) -> &[f32] {
    &slab[INDEX as usize * D..(INDEX as usize + 1) * D]
}

fn main() -> Result<(), Box<dyn Error>> {
    let values: Vec<f32> = (0..D).map(|v| v as f32 + 1.0).collect();
    let row = api::DeviceOpReshape::reshape(
        api::copy_host_vec_to_device(&Arc::new(values.clone())),
        &[1, D],
    )
    .sync()?;

    match compile_dynamic_ir() {
        Ok(ir) => println!("write_row_dynamic Tile IR: ok ({} bytes)", ir.len()),
        Err(error) => println!("write_row_dynamic Tile IR: ERROR {error}"),
    }

    let mut slab = api::zeros::<f32>(&[ROWS, D]).sync()?;
    kernels::write_row_static((&mut slab).partition([ROWS, D]), &row, INDEX)
        .generics(vec![ROWS.to_string(), D.to_string()])
        .sync()?;
    let host = slab.dup().to_host_vec().sync()?;
    println!(
        "write_row_static launch: ok, row {INDEX} written: {}",
        written_row(&host) == values.as_slice()
    );

    let mut slab = api::zeros::<f32>(&[ROWS, D]).sync()?;
    let launched = kernels::write_row_dynamic((&mut slab).partition([ROWS, D]), &row, INDEX)
        .generics(vec![D.to_string()])
        .sync();
    match launched {
        Ok(_) => {
            let host = slab.dup().to_host_vec().sync()?;
            println!(
                "write_row_dynamic launch: ok, row {INDEX} written: {}",
                written_row(&host) == values.as_slice()
            );
        }
        Err(error) => println!("write_row_dynamic launch: ERROR {error}"),
    }
    Ok(())
}
