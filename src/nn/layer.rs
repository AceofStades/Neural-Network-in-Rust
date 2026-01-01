use crate::nn::activation;
use ndarray::{Array1, Array2};

pub fn forward_prop(a_prev: Array1<f32>, w: Array2<f32>, b: Array1<f32>) -> Array1<f32> {
    let z: Array1<f32> = a_prev.dot(w[1]) + b;
    a_prev
}
