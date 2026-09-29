use firepit::{Engine, Model};
use std::sync::Arc;
fn mutate(engine: &mut Engine, model: Arc<Model>) { engine.model = model; }
fn main() {}
