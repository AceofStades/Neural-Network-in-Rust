use std::f64::consts;

const E: f32 = consts::E as f32;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

pub fn tanh(x: f32) -> f32 {
    2.0 / (1.0 + E.powf(-2.0 * x)) - 1.0
}

pub fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 as f32 }
}

// pub fn softmax(x: f32) -> f32 {}

pub fn softplus(x: f32) -> f32 {
    (1.0 + E.powf(x)).log(E)
}
