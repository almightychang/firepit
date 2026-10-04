use cuda_async::cuda_graph::Scope;
use cuda_async::device_operation::{DeviceOp, GraphNode};
use cuda_async::error::DeviceError;
use std::collections::BTreeMap;
use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

pub trait Sink {
    fn run<T: Send, O: GraphNode + DeviceOp<Output = T>>(
        &mut self,
        op: O,
    ) -> Result<(), DeviceError>;

    fn span<T>(
        &mut self,
        _kind: &'static str,
        f: impl FnOnce(&mut Self) -> Result<T, Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>> {
        f(self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    pub kind: &'static str,
    pub gpu_ms: f64,
    pub wall_ms: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trace {
    pub spans: Vec<Span>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    pub kind: &'static str,
    pub count: usize,
    pub gpu_ms: f64,
    pub wall_ms: f64,
}

impl Trace {
    pub fn summary(&self) -> Vec<Summary> {
        let mut totals = BTreeMap::<&'static str, Summary>::new();
        for span in &self.spans {
            let row = totals.entry(span.kind).or_insert(Summary {
                kind: span.kind,
                count: 0,
                gpu_ms: 0.0,
                wall_ms: 0.0,
            });
            row.count += 1;
            row.gpu_ms += span.gpu_ms;
            row.wall_ms += span.wall_ms;
        }
        totals.into_values().collect()
    }
}

pub struct Traced<'a> {
    stream: &'a Arc<cuda_core::Stream>,
    trace: Trace,
    measured_gpu_ms: f64,
}

impl<'a> Traced<'a> {
    pub fn new(stream: &'a Arc<cuda_core::Stream>) -> Self {
        Self {
            stream,
            trace: Trace::default(),
            measured_gpu_ms: 0.0,
        }
    }

    pub fn into_trace(self) -> Trace {
        self.trace
    }
}

impl Sink for Traced<'_> {
    fn run<T: Send, O: GraphNode + DeviceOp<Output = T>>(
        &mut self,
        op: O,
    ) -> Result<(), DeviceError> {
        let start = self.stream.device().new_event()?;
        let end = self.stream.device().new_event()?;
        start.record(self.stream)?;
        op.sync_on(self.stream)?;
        end.record(self.stream)?;
        end.synchronize()?;
        self.measured_gpu_ms += start.elapsed_time(&end)? as f64;
        Ok(())
    }

    fn span<T>(
        &mut self,
        kind: &'static str,
        f: impl FnOnce(&mut Self) -> Result<T, Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>> {
        let wall = Instant::now();
        let gpu_before = self.measured_gpu_ms;
        let value = f(self)?;
        self.trace.spans.push(Span {
            kind,
            gpu_ms: self.measured_gpu_ms - gpu_before,
            wall_ms: wall.elapsed().as_secs_f64() * 1e3,
        });
        Ok(value)
    }
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
