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

pub fn channel_tile(channels: usize, sm_count: usize, blocks_per_sm: usize) -> usize {
    let largest_divisor = 1usize << channels.trailing_zeros();
    row_tile(
        largest_divisor,
        channels / largest_divisor,
        sm_count,
        blocks_per_sm,
        1,
    )
}

#[cfg(test)]
mod tests {
    use super::{channel_tile, row_tile};

    #[test]
    fn responds_to_device_size() {
        assert_eq!(row_tile(128, 8, 84, 1, 4), 8);
        assert_eq!(row_tile(128, 8, 48, 1, 4), 16);
        assert_eq!(row_tile(128, 8, 84, 2, 4), 4);
        assert_eq!(row_tile(128, 8, 84, 4, 8), 8);
    }

    #[test]
    fn channel_tiles_cover_all_channels() {
        assert_eq!(channel_tile(8192, 84, 1), 64);
        assert_eq!(channel_tile(8192, 48, 1), 128);
        assert_eq!(channel_tile(96, 84, 1), 1);
        assert_eq!(channel_tile(96, 8, 1), 8);
    }
}
