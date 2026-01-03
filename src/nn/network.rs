use crate::nn::activation::*;
use ndarray::{Array, Array1, Array2};

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

    pub last_input: Option<Array1<f32>>,
    pub last_z: Option<Array1<f32>>,
}

impl Layer {
    pub fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        self.last_input = Some(input.clone());

        let z = input.dot(&self.weights) + &self.biases;
        self.last_z = Some(z.clone());

        match self.activation {
            ActivationType::ReLU => z.mapv(|x| relu(x)),
            ActivationType::Sigmoid => z.mapv(|x| sigmoid(x)),
            ActivationType::SoftPlus => z.mapv(|x| softplus(x)),
            ActivationType::Tanh => z.mapv(|x| tanh(x)),
        }
    }

    pub fn backward(&mut self, output_grad: Array1<f32>) {}
}
