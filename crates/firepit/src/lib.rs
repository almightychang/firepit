mod engine;
mod model;
mod plan;
mod session;

pub use engine::{Engine, Execution};
pub use model::{Model, ModelConfig};
pub use plan::{Op, SessionKind, Slot, StepPlan};
pub use session::{Batch, LayerState, Session, SessionId};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
