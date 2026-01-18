use crate::nn::activation::*;
use ndarray::{Array1, Array2, Axis};

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

    last_input: Option<Array1<f32>>,
    last_z: Option<Array1<f32>>,
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

    pub fn backward(&mut self, output_grad: Array1<f32>, learning_rate: f32) -> Array1<f32> {
        let z = self.last_z.as_ref().unwrap();

        let grad_fn = match self.activation {
            ActivationType::ReLU => z.mapv(|x| relu_grad(x)),
            ActivationType::Sigmoid => z.mapv(|x| sigmoid_grad(x)),
            ActivationType::SoftPlus => z.mapv(|x| softplus_grad(x)),
            ActivationType::Tanh => z.mapv(|x| tanh_grad(x)),
        };

        let dz = output_grad * grad_fn;

        let input = self.last_input.as_ref().unwrap();
        let input_col = input.view().insert_axis(Axis(1));
        let dz_row = dz.view().insert_axis(Axis(0));
        let dw = input_col.dot(&dz_row);

        let da_prev = dz.dot(&self.weights.t());

        self.weights = &self.weights - &(learning_rate * &dw);
        self.biases = &self.biases - &(learning_rate * &dz);

        da_prev
    }
}
