pub const MIN_ROW_TILE: usize = 4;

pub fn splits_into_row_tiles(v: usize) -> bool {
    v.is_power_of_two() && v >= MIN_ROW_TILE
}

pub fn row_tile(
    v: usize,
    heads: usize,
    sm_count: usize,
    blocks_per_sm: usize,
    min_tile: usize,
) -> usize {
    let target = sm_count.saturating_mul(blocks_per_sm);
    let mut tile = v;
    while tile > min_tile && heads * (v / tile) < target {
        tile /= 2;
    }
    tile
}

#[cfg(test)]
mod tests {
    use super::row_tile;

    #[test]
    fn responds_to_device_size() {
        assert_eq!(row_tile(128, 8, 84, 1, 4), 8);
        assert_eq!(row_tile(128, 8, 48, 1, 4), 16);
        assert_eq!(row_tile(128, 8, 84, 2, 4), 4);
        assert_eq!(row_tile(128, 8, 84, 4, 8), 8);
    }
}
