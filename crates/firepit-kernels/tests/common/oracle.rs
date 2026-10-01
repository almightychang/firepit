use half::f16;

pub fn dense(weights: &[f32], input: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    assert_eq!(weights.len(), rows * cols);
    assert_eq!(input.len() % cols, 0);
    let mut output = Vec::with_capacity(input.len() / cols * rows);
    for x in input.chunks_exact(cols) {
        for w in weights.chunks_exact(cols) {
            output.push(w.iter().zip(x).map(|(&w, &x)| w as f64 * x as f64).sum());
        }
    }
    output
}

pub fn q8(qs: &[i8], scales: &[f16], input: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    assert_eq!(qs.len(), rows * cols);
    assert_eq!(scales.len(), rows * cols / 32);
    let weights: Vec<f32> = qs
        .iter()
        .enumerate()
        .map(|(i, &q)| q as f32 * scales[i / 32].to_f32())
        .collect();
    dense(&weights, input, rows, cols)
}

pub fn rms_norm(input: &[f32], gamma: &[f32], eps: f32) -> Vec<f64> {
    let width = gamma.len();
    assert_eq!(input.len() % width, 0);
    input
        .chunks_exact(width)
        .flat_map(|row| {
            let mean_square = row.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / width as f64;
            let inv = (mean_square + eps as f64).sqrt().recip();
            row.iter()
                .zip(gamma)
                .map(move |(&x, &g)| x as f64 * inv * g as f64)
        })
        .collect()
}

pub fn swiglu(gate_up: &[f32], width: usize) -> Vec<f64> {
    assert_eq!(gate_up.len() % (2 * width), 0);
    gate_up
        .chunks_exact(2 * width)
        .flat_map(|row| {
            let (gate, up) = row.split_at(width);
            gate.iter().zip(up).map(|(&g, &u)| {
                let g = g as f64;
                let sigmoid = if g >= 0.0 {
                    1.0 / (1.0 + (-g).exp())
                } else {
                    g.exp() / (1.0 + g.exp())
                };
                g * sigmoid * u as f64
            })
        })
        .collect()
}
