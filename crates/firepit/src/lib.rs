mod access;
mod checkpoint;
mod config;
mod engine;
mod execute;
pub mod generation;
mod model;
mod plan;
mod prefill;
pub mod quality;
mod session;
#[cfg(feature = "testing")]
mod testing;

pub use checkpoint::{Checkpoint, HostBf16Tensor, HostI32Tensor, HostTensor, SafetensorsFile};
pub use config::{LayerKind, ModelConfig, ModelDims, WeightFormat};
pub use engine::Engine;
#[cfg(feature = "testing")]
pub use engine::{IsolatedLayerTrace, RoutingLayerTrace, SessionStateSnapshot, VerificationFault};
pub use model::{LoadOptions, Model, WeightGroup, WeightOverride, WeightReport, verify_checkpoint};
pub use prefill::ScoredRows;
pub use safetensors::Dtype;
pub use session::{Batch, LayerState, Session, SessionId};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
