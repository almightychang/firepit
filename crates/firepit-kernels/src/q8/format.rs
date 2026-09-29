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
