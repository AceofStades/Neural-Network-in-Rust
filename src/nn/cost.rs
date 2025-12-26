pub fn mse(y_true: Vec<f32>, y_pred: Vec<f32>) -> f32 {
    assert_eq!(y_true.len(), y_pred.len());
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(y, y_hat)| (y - y_hat).powi(2))
        .sum::<f32>()
        / y_true.len() as f32
}

pub fn mae(y_true: Vec<f32>, y_pred: Vec<f32>) -> f32 {
    assert_eq!(y_true.len(), y_pred.len());
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(y, y_hat)| (y - y_hat).abs())
        .sum::<f32>()
        / y_true.len() as f32
}

pub fn mbe(y_true: Vec<f32>, y_pred: Vec<f32>) -> f32 {
    assert_eq!(y_true.len(), y_pred.len());
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(y, y_hat)| y - y_hat)
        .sum::<f32>()
}

pub fn huber(y_true: Vec<f32>, y_pred: Vec<f32>, delta: f32) -> f32 {
    assert_eq!(y_true.len(), y_pred.len());
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(y, y_hat)| {
            let alpha: f32 = (y - y_hat).abs();

            if alpha <= delta {
                1.0 / 2.0 * alpha.powi(2)
            } else {
                delta * (alpha - 1.0 / 2.0 * delta)
            }
        })
        .sum::<f32>()
        / y_true.len() as f32
}

pub fn cce(y_true: Vec<f32>, y_pred: Vec<f32>) -> f32 {
    assert_eq!(y_true.len(), y_pred.len());
    -y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(y, y_hat)| y * y_hat.ln())
        .sum::<f32>()
}

pub fn bce(y_true: Vec<f32>, y_pred: Vec<f32>) -> f32 {
    assert_eq!(y_true.len(), y_pred.len());
    -y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(y, y_hat)| y * y_hat.ln() + (1.0 - y) * (1.0 - y_hat).ln())
        .sum::<f32>()
        / y_true.len() as f32
}
