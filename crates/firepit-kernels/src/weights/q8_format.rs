use half::f16;

pub const Q8_BLOCK: usize = 32;
const GGUF_Q8_0_BLOCK_BYTES: usize = std::mem::size_of::<u16>() + Q8_BLOCK;

fn shape(rows: usize, cols: usize) -> Result<(usize, usize), FormatError> {
    if rows == 0 || cols == 0 || cols % Q8_BLOCK != 0 {
        return Err(FormatError::InvalidShape);
    }
    let elements = rows.checked_mul(cols).ok_or(FormatError::InvalidShape)?;
    Ok((elements, elements / Q8_BLOCK))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    InvalidShape,
    InvalidLength { expected: usize, actual: usize },
    InvalidInputLength { expected: usize, actual: usize },
}

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for FormatError {}

pub struct HostQ8Matrix {
    pub(crate) qs: Vec<i8>,
    pub(crate) scales: Vec<f16>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl HostQ8Matrix {
    pub fn from_parts(
        qs: Vec<i8>,
        scales: Vec<f16>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, FormatError> {
        let (elements, blocks) = shape(rows, cols)?;
        if qs.len() != elements {
            return Err(FormatError::InvalidLength {
                expected: elements,
                actual: qs.len(),
            });
        }
        if scales.len() != blocks {
            return Err(FormatError::InvalidLength {
                expected: blocks,
                actual: scales.len(),
            });
        }
        Ok(Self {
            qs,
            scales,
            rows,
            cols,
        })
    }
    pub fn from_gguf_q8_0(blocks: &[u8], rows: usize, cols: usize) -> Result<Self, FormatError> {
        let (elements, count) = shape(rows, cols)?;
        let expected = count
            .checked_mul(GGUF_Q8_0_BLOCK_BYTES)
            .ok_or(FormatError::InvalidShape)?;
        if blocks.len() != expected {
            return Err(FormatError::InvalidLength {
                expected,
                actual: blocks.len(),
            });
        }
        let mut qs = Vec::with_capacity(elements);
        let mut scales = Vec::with_capacity(count);
        for block in blocks.chunks_exact(GGUF_Q8_0_BLOCK_BYTES) {
            scales.push(f16::from_bits(u16::from_le_bytes([block[0], block[1]])));
            qs.extend(block[std::mem::size_of::<u16>()..].iter().map(|&q| q as i8));
        }
        Self::from_parts(qs, scales, rows, cols)
    }

    pub fn gemv_reference(&self, x: &[f32]) -> Result<Vec<f64>, FormatError> {
        if x.len() % self.cols != 0 {
            return Err(FormatError::InvalidInputLength {
                expected: self.cols,
                actual: x.len(),
            });
        }
        let batch = x.len() / self.cols;
        let mut y = vec![0.0; batch * self.rows];
        for b in 0..batch {
            for n in 0..self.rows {
                let mut sum = 0.0f64;
                for k in 0..self.cols {
                    let scale =
                        self.scales[n * (self.cols / Q8_BLOCK) + k / Q8_BLOCK].to_f32() as f64;
                    sum += self.qs[n * self.cols + k] as f64 * scale * x[b * self.cols + k] as f64;
                }
                y[b * self.rows + n] = sum;
            }
        }
        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn gguf_decodes_every_generated_block(
            case in (1usize..5, 1usize..5).prop_flat_map(|(rows, blocks)| {
                (Just(rows), Just(blocks),
                 prop::collection::vec(any::<u8>(), rows * blocks * GGUF_Q8_0_BLOCK_BYTES))
            })
        ) {
            let (rows, blocks_per_row, bytes) = case;
            let result = HostQ8Matrix::from_gguf_q8_0(
                &bytes, rows, blocks_per_row * Q8_BLOCK,
            );
            prop_assert!(result.is_ok());
            let matrix = result.unwrap();
            for (block_index, block) in bytes.chunks_exact(GGUF_Q8_0_BLOCK_BYTES).enumerate() {
                let scale_bits = u16::from_le_bytes([block[0], block[1]]);
                prop_assert_eq!(matrix.scales[block_index].to_bits(), scale_bits);
                for (offset, &quant) in block[2..].iter().enumerate() {
                    prop_assert_eq!(matrix.qs[block_index * Q8_BLOCK + offset], quant as i8);
                }
            }
        }

        #[test]
        fn gguf_length_contract(
            case in (1usize..5, 1usize..5, -2isize..3).prop_flat_map(|(rows, blocks, delta)| {
                let expected = rows * blocks * GGUF_Q8_0_BLOCK_BYTES;
                (Just(rows), Just(blocks), Just(expected),
                 prop::collection::vec(any::<u8>(), (expected as isize + delta) as usize))
            })
        ) {
            let (rows, blocks_per_row, expected, bytes) = case;
            let result = HostQ8Matrix::from_gguf_q8_0(
                &bytes, rows, blocks_per_row * Q8_BLOCK,
            );
            if bytes.len() == expected {
                prop_assert!(result.is_ok());
            } else {
                prop_assert_eq!(result.err(), Some(FormatError::InvalidLength {
                    expected,
                    actual: bytes.len(),
                }));
            }
        }

        #[test]
        fn parts_length_contract(
            rows in 1usize..5,
            blocks_per_row in 1usize..5,
            quant_delta in -1isize..2,
            scale_delta in -1isize..2,
        ) {
            let elements = rows * blocks_per_row * Q8_BLOCK;
            let blocks = rows * blocks_per_row;
            let quant_len = (elements as isize + quant_delta) as usize;
            let scale_len = (blocks as isize + scale_delta) as usize;
            let result = HostQ8Matrix::from_parts(
                vec![0; quant_len],
                vec![f16::from_f32(1.0); scale_len],
                rows,
                blocks_per_row * Q8_BLOCK,
            );
            let expected = if quant_len != elements {
                Some(FormatError::InvalidLength { expected: elements, actual: quant_len })
            } else if scale_len != blocks {
                Some(FormatError::InvalidLength { expected: blocks, actual: scale_len })
            } else {
                None
            };
            prop_assert_eq!(result.err(), expected);
        }

        #[test]
        fn malformed_shapes_are_rejected(rows in 0usize..5, cols in 0usize..100) {
            if rows == 0 || cols == 0 || cols % Q8_BLOCK != 0 {
                prop_assert_eq!(
                    HostQ8Matrix::from_gguf_q8_0(&[], rows, cols).err(),
                    Some(FormatError::InvalidShape),
                );
            }
        }
    }

    #[test]
    fn rejects_shape_overflow() {
        assert_eq!(
            HostQ8Matrix::from_gguf_q8_0(&[], usize::MAX, Q8_BLOCK).err(),
            Some(FormatError::InvalidShape),
        );
    }

    #[test]
    fn parses_and_multiplies() {
        let mut bytes = Vec::new();
        for scale in [0.5f32, -0.25] {
            bytes.extend(f16::from_f32(scale).to_bits().to_le_bytes());
            bytes.extend((0..32).map(|i| (i as i8 - 16) as u8));
        }
        let w = HostQ8Matrix::from_gguf_q8_0(&bytes, 1, 64).unwrap();
        assert_eq!(w.qs[0], -16);
        assert_eq!(w.qs[63], 15);
        assert_eq!(w.gemv_reference(&vec![1.0; 128]).unwrap(), vec![-4.0, -4.0]);
    }

    #[test]
    fn rejects_bad_parts() {
        assert!(matches!(
            HostQ8Matrix::from_parts(vec![0; 31], vec![f16::from_f32(1.0)], 1, 32),
            Err(FormatError::InvalidLength {
                expected: 32,
                actual: 31
            })
        ));
        assert!(matches!(
            HostQ8Matrix::from_parts(vec![0; 32], vec![], 1, 32),
            Err(FormatError::InvalidLength {
                expected: 1,
                actual: 0
            })
        ));
    }

    #[test]
    fn rejects_bad_shapes() {
        assert_eq!(
            HostQ8Matrix::from_gguf_q8_0(&[], 1, 33).err(),
            Some(FormatError::InvalidShape)
        );
        assert_eq!(
            HostQ8Matrix::from_gguf_q8_0(&[], 1, 32).err(),
            Some(FormatError::InvalidLength {
                expected: 34,
                actual: 0
            })
        );
    }
}
