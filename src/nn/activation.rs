use std::f64::consts;

fn sigmoid(x: f32) -> f32 {
    let e: f32 = consts::E as f32;
    1.0 / (1.0 + e.powf(-x))
}

fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 as f32 }
}

fn softmax(x: f32) -> f32 {}
