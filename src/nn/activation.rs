use std::f64::consts;

const E: f32 = consts::E as f32;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

pub fn sigmoid_grad(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub fn tanh(x: f32) -> f32 {
    2.0 / (1.0 + E.powf(-2.0 * x)) - 1.0
}

pub fn tanh_grad(x: f32) -> f32 {
    1.0 - tanh(x).powi(2)
}

pub fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

pub fn relu_grad(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

// pub fn softmax(x: f32) -> f32 {}

pub fn softplus(x: f32) -> f32 {
    (1.0 + E.powf(x)).log(E)
}

pub fn softplus_grad(x: f32) -> f32 {
    sigmoid(x)
}

pub fn linear(x: f32) -> f32 {
    x
}

pub fn linear_grad(_x: f32) -> f32 {
    1.0
}
