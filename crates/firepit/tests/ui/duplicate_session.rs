use firepit::{Batch, Session};

fn duplicate(session: &mut Session) {
    let _batch = Batch::new(vec![&mut *session, &mut *session], vec![1, 2]);
}

fn main() {}
