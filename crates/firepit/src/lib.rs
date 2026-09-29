mod engine;
mod model;
mod plan;
mod session;

pub use engine::Engine;
pub use model::{DimensionError, Model, ModelConfig, ModelDims};
pub use session::{Batch, Session, SessionId};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
