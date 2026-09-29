use firepit::{Batch, Session};
fn make(session: &mut Session) { let _ = Batch { sessions: vec![session], tokens: vec![1], original_rows: vec![0] }; }
fn main() {}
