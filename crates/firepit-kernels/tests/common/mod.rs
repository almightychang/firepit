#[allow(dead_code)]
pub mod check;
#[allow(dead_code)]
pub mod oracle;
#[allow(dead_code)]
pub mod strategy;

use cuda_core::{Device, Stream};
use std::sync::Arc;

pub fn stream() -> Arc<Stream> {
    let device = Device::new(0).unwrap();
    device.new_stream().unwrap()
}

pub fn runner(cases: u32, source_file: &'static str) -> proptest::test_runner::TestRunner {
    proptest::test_runner::TestRunner::new(proptest::test_runner::Config {
        cases,
        source_file: Some(source_file),
        failure_persistence: Some(Box::new(
            proptest::test_runner::FileFailurePersistence::WithSource("proptest-regressions"),
        )),
        ..Default::default()
    })
}
