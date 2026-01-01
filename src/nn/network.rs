use crate::nn::activation::*;
use ndarray::{Array1, Array2};

pub enum ActivationType {
    Sigmoid,
    ReLU,
    Tanh,
    SoftPlus,
}

pub struct Layer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub activation: ActivationType,
}

impl Layer {
    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let z = input.dot(&self.weights) + &self.biases;
        match self.activation {
            ActivationType::ReLU => z.mapv(|x| relu(x)),
            ActivationType::Sigmoid => z.mapv(|x| sigmoid(x)),
            ActivationType::SoftPlus => z.mapv(|x| softplus(x)),
            ActivationType::Tanh => z.mapv(|x| tanh(x)),
        }
    }
}
