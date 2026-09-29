#[test]
fn same_session_cannot_appear_twice() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/duplicate_session.rs");
}
