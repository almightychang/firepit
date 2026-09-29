#[test]
fn public_invariants_cannot_be_bypassed() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/*.rs");
}
