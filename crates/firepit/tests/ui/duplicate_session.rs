use firepit::{Engine, Session};

fn duplicate(engine: &Engine, session: &mut Session) {
    let _batch = engine.batch([(&mut *session, 1), (&mut *session, 2)]);
}
fn main() {}
