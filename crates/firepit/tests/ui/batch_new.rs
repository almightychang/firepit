use firepit::{Batch, Session};
fn make(session: &mut Session) { let _ = Batch::new(vec![session], vec![1]); }
fn main() {}
