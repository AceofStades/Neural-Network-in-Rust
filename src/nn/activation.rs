use std::f64::consts;

fn sigmoid(x: f32) -> f64 {
    let e: f64 = consts::E;
    1.0 / (1.0 + e.powf(x as f64))
}
