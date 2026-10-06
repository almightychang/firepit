use cutile::cutile_compiler::specialization::DivHint;
use cutile::jit_cache::{self, FileSystemJitStore};
use cutile::tile_kernel::jit_compile_count;
use firepit::Result;
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::Path;
use std::sync::Arc;

/// Keeps compiled cubins in `dir` across runs. A cache hit loads stored bytes as
/// device code, so `dir` must be private: it is created 0700, and an existing one
/// is refused when group or others can access it. Restoring 0700 on an existing
/// directory succeeds only for its owner.
pub fn enable_disk_cache(dir: &Path) -> Result<()> {
    if let Some(parent) = dir.parent() {
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(parent)?;
    }
    match std::fs::DirBuilder::new().mode(0o700).create(dir) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let metadata = std::fs::symlink_metadata(dir)?;
            let mode = metadata.permissions().mode();
            if !metadata.is_dir() || mode & 0o077 != 0 {
                return Err(format!(
                    "refusing JIT cache {}: not a private directory (mode {:04o}); delete it and retry",
                    dir.display(),
                    mode & 0o777
                )
                .into());
            }
        }
        Err(error) => return Err(error.into()),
    }
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    jit_cache::enable(Arc::new(FileSystemJitStore::new(dir)?));
    Ok(())
}

/// KV cache length for `tokens`, capped at `limit`. Decode `cache_append` takes
/// the cache length as a compile-time slab shape (cuTile requires static shapes
/// on written tensors), so lengths are rounded to powers of two, no smaller than
/// the largest shape divisibility cuTile specializes on: logarithmically many
/// compiled variants, at most twice the cache memory the tokens need.
pub fn cache_capacity(tokens: usize, limit: usize) -> usize {
    tokens
        .next_power_of_two()
        .max(DivHint::default().max as usize)
        .min(limit)
}

#[derive(Clone, Copy)]
pub struct Counts {
    compiled: u64,
    tileiras: u64,
    disk_hits: u64,
}

impl Counts {
    pub fn now() -> Self {
        Self {
            compiled: jit_compile_count(),
            tileiras: jit_cache::jit_backend_compile_count(),
            disk_hits: jit_cache::jit_disk_hit_count(),
        }
    }

    pub fn since(self, start: Self) -> String {
        format!(
            "jit: {} kernels compiled, {} by tileiras, {} from the disk cache",
            self.compiled - start.compiled,
            self.tileiras - start.tileiras,
            self.disk_hits - start.disk_hits
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("firepit-jit-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn mode(path: &Path) -> u32 {
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[test]
    fn creates_a_private_cache_directory() -> Result<()> {
        let home = scratch("new");
        let dir = home.join("jit");
        enable_disk_cache(&dir)?;
        assert_eq!(mode(&dir), 0o700);
        jit_cache::disable();
        std::fs::remove_dir_all(home)?;
        Ok(())
    }

    #[test]
    fn refuses_a_cache_directory_others_can_read() -> Result<()> {
        let home = scratch("shared");
        let dir = home.join("jit");
        std::fs::create_dir_all(&dir)?;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755))?;
        assert!(enable_disk_cache(&dir).is_err());
        assert_eq!(mode(&dir), 0o755);
        std::fs::remove_dir_all(home)?;
        Ok(())
    }

    #[test]
    fn cache_capacity_rounds_to_powers_of_two_within_the_limit() {
        let floor = DivHint::default().max as usize;
        assert_eq!(cache_capacity(1, 1 << 18), floor);
        assert_eq!(cache_capacity(floor + 1, 1 << 18), 2 * floor);
        assert_eq!(cache_capacity(100, 1 << 18), 128);
        assert_eq!(cache_capacity(128, 1 << 18), 128);
        assert_eq!(cache_capacity(3000, 2048 + 1000), 3048);
        let distinct: std::collections::BTreeSet<_> = (1..=4096)
            .map(|tokens| cache_capacity(tokens, 1 << 18))
            .collect();
        assert_eq!(distinct.len(), (4096 / floor).ilog2() as usize + 1);
    }
}
