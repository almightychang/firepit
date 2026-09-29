use half::f16;

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
    pub qs: Vec<i8>,
    pub scales: Vec<f16>,
    pub rows: usize,
    pub cols: usize,
}

pub struct HostQuantizedX {
    pub xq: Vec<i8>,
    pub dx: Vec<f32>,
    pub batch: usize,
    pub cols: usize,
}

pub fn quantize_x_cpu(x: &[f32], batch: usize, cols: usize) -> Result<HostQuantizedX, FormatError> {
    if batch == 0 || cols == 0 || cols % 32 != 0 {
        return Err(FormatError::InvalidShape);
    }
    let expected = batch.checked_mul(cols).ok_or(FormatError::InvalidShape)?;
    if x.len() != expected {
        return Err(FormatError::InvalidInputLength {
            expected,
            actual: x.len(),
        });
    }
    let mut xq = Vec::with_capacity(expected);
    let mut dx = Vec::with_capacity(expected / 32);
    for block in x.chunks_exact(32) {
        let max_abs = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = max_abs / 127.0;
        dx.push(scale);
        let divisor = if scale == 0.0 { 1.0 } else { scale };
        xq.extend(
            block
                .iter()
                .map(|&v| (v / divisor).round().clamp(-127.0, 127.0) as i8),
        );
    }
    Ok(HostQuantizedX {
        xq,
        dx,
        batch,
        cols,
    })
}

impl HostQ8Matrix {
    pub fn from_gguf_q8_0(blocks: &[u8], rows: usize, cols: usize) -> Result<Self, FormatError> {
        if rows == 0 || cols == 0 || cols % 32 != 0 {
            return Err(FormatError::InvalidShape);
        }
        let count = rows
            .checked_mul(cols / 32)
            .ok_or(FormatError::InvalidShape)?;
        let expected = count.checked_mul(34).ok_or(FormatError::InvalidShape)?;
        if blocks.len() != expected {
            return Err(FormatError::InvalidLength {
                expected,
                actual: blocks.len(),
            });
        }
        let mut qs = Vec::with_capacity(rows * cols);
        let mut scales = Vec::with_capacity(count);
        for block in blocks.chunks_exact(34) {
            scales.push(f16::from_bits(u16::from_le_bytes([block[0], block[1]])));
            qs.extend(block[2..].iter().map(|&q| q as i8));
        }
        Ok(Self {
            qs,
            scales,
            rows,
            cols,
        })
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
                    let scale = self.scales[n * (self.cols / 32) + k / 32].to_f32() as f64;
                    sum += self.qs[n * self.cols + k] as f64 * scale * x[b * self.cols + k] as f64;
                }
                y[b * self.rows + n] = sum;
            }
        }
        Ok(y)
    }

    pub fn gemv_quantized_reference(&self, x: &HostQuantizedX) -> Result<Vec<f64>, FormatError> {
        if x.cols != self.cols {
            return Err(FormatError::InvalidInputLength {
                expected: self.cols,
                actual: x.cols,
            });
        }
        let blocks = self.cols / 32;
        let mut y = vec![0.0f64; x.batch * self.rows];
        for b in 0..x.batch {
            for n in 0..self.rows {
                let mut sum = 0.0f64;
                for block in 0..blocks {
                    let base_w = (n * blocks + block) * 32;
                    let base_x = (b * blocks + block) * 32;
                    let dot: i32 = (0..32)
                        .map(|i| self.qs[base_w + i] as i32 * x.xq[base_x + i] as i32)
                        .sum();
                    let scale = self.scales[n * blocks + block].to_f32() as f64;
                    sum += dot as f64 * scale * x.dx[b * blocks + block] as f64;
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

    #[test]
    fn quantizes_each_block() {
        let x = [
            vec![0.0f32; 32],
            (0..32).map(|i| (i as f32 - 16.0) / 4.0).collect(),
        ]
        .concat();
        let q = quantize_x_cpu(&x, 1, 64).unwrap();
        assert_eq!(&q.xq[..32], &[0; 32]);
        assert_eq!(q.dx[0], 0.0);
        assert_eq!(q.xq[32], -127);
        assert_eq!(q.dx[1], 4.0 / 127.0);
    }
}
