use cuda_async::cuda_graph::Scope;
use cuda_async::device_operation::{DeviceOp, GraphNode};
use cuda_async::error::DeviceError;
use std::sync::Arc;

pub trait Sink {
    fn run<T: Send, O: GraphNode + DeviceOp<Output = T>>(
        &mut self,
        op: O,
    ) -> Result<(), DeviceError>;
}

pub struct Eager<'a>(pub &'a Arc<cuda_core::Stream>);

impl Sink for Eager<'_> {
    fn run<T: Send, O: GraphNode + DeviceOp<Output = T>>(
        &mut self,
        op: O,
    ) -> Result<(), DeviceError> {
        op.sync_on(self.0)?;
        Ok(())
    }
}

pub struct Graph<'a>(pub &'a Scope);

impl Sink for Graph<'_> {
    fn run<T: Send, O: GraphNode + DeviceOp<Output = T>>(
        &mut self,
        op: O,
    ) -> Result<(), DeviceError> {
        self.0.record(op)?;
        Ok(())
    }
}
