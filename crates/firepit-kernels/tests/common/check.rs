pub fn max_relative(actual: &[f32], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(&a, &e)| (a as f64 - e).abs() / e.abs().max(1.0))
        .fold(0.0, f64::max)
}

pub fn max_abs(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

const UNIT_ROUNDOFF: f64 = f32::EPSILON as f64 / 2.0;
const LAMBDA: f64 = 8.0;

// Higham & Mary (2019), Theorem 3.1: |ŷ − y| ≤ γ̃_n(λ) · |a|ᵀ|b|, with probability ≥ 1 − n(1 − P(λ)).
pub fn gamma_tilde(n: usize) -> f64 {
    let (n, u) = (n as f64, UNIT_ROUNDOFF);
    (LAMBDA * n.sqrt() * u + n * u * u / (1.0 - u)).exp_m1()
}

pub fn inner_product_excess(actual: &[f32], expected: &[f64], magnitude: &[f64], n: usize) -> f64 {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.len(), magnitude.len());
    let bound = gamma_tilde(n);
    actual
        .iter()
        .zip(expected)
        .zip(magnitude)
        .map(|((&a, &e), &m)| (a as f64 - e).abs() / (bound * m).max(f64::MIN_POSITIVE))
        .fold(0.0, f64::max)
}

// An activation φ after the inner product moves its error by at most max|φ'|, plus its own rounding.
pub fn activated_inner_product_excess(
    actual: &[f32],
    expected: &[f64],
    magnitude: &[f64],
    n: usize,
    max_slope: f64,
) -> f64 {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.len(), magnitude.len());
    let bound = gamma_tilde(n);
    actual
        .iter()
        .zip(expected)
        .zip(magnitude)
        .map(|((&a, &e), &m)| {
            let allowed = max_slope * bound * m + 8.0 * UNIT_ROUNDOFF * e.abs();
            (a as f64 - e).abs() / allowed.max(f64::MIN_POSITIVE)
        })
        .fold(0.0, f64::max)
}
