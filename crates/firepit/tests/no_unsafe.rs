use std::fs;
use std::path::{Path, PathBuf};

fn visit_rs_files(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            visit_rs_files(&path, files);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            files.push(path);
        }
    }
}

#[test]
fn engine_and_kernel_sources_have_no_unsafe() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut files = Vec::new();
    for crate_name in ["firepit", "firepit-kernels"] {
        visit_rs_files(
            &workspace.join("crates").join(crate_name).join("src"),
            &mut files,
        );
    }
    files.sort();

    let mut violations = Vec::new();
    for file in files {
        let source = fs::read_to_string(&file).unwrap();
        for (line_number, line) in source.lines().enumerate() {
            if line
                .split(|ch: char| !ch.is_alphanumeric() && ch != '_')
                .any(|token| token == "unsafe")
            {
                violations.push(format!(
                    "{}:{}",
                    file.strip_prefix(&workspace).unwrap().display(),
                    line_number + 1
                ));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "unsafe in source:\n{}",
        violations.join("\n")
    );
}
