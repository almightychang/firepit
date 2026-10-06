use std::path::Path;

#[test]
fn decode_runner_reads_weight_matrices_through_the_accessor() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/execute.rs");
    let source = std::fs::read_to_string(&path).unwrap();
    let violations: Vec<String> = source
        .lines()
        .enumerate()
        .filter(|(_, line)| {
            let direct = line
                .match_indices("weight(")
                .any(|(at, _)| !line[..at].ends_with(|ch: char| ch.is_alphanumeric() || ch == '_'));
            let unnamed_matrix = line
                .match_indices(".matrix(")
                .any(|(at, _)| !line[..at].ends_with("weights"));
            direct || unnamed_matrix || line.contains(".moe.gate_up") || line.contains(".moe.down")
        })
        .map(|(number, line)| format!("{}: {}", number + 1, line.trim()))
        .collect();
    assert!(
        violations.is_empty(),
        "execute.rs reaches weight matrices around Weights::matrix:\n{}",
        violations.join("\n")
    );
}
