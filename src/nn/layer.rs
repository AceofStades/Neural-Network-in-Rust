use crate::nn::activation::*;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum ActivationType {
    Sigmoid,
    ReLU,
    Tanh,
    SoftPlus,
    Linear,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Layer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub activation: ActivationType,

    #[serde(skip)]
    last_input: Option<Array1<f32>>,
    #[serde(skip)]
    last_z: Option<Array1<f32>>,
    
    #[serde(skip)]
    accumulated_dw: Option<Array2<f32>>,
    #[serde(skip)]
    accumulated_db: Option<Array1<f32>>,
    #[serde(skip)]
    batch_count: usize,
}

impl Layer {
    pub fn new(inputs: usize, outputs: usize, activation: ActivationType) -> Self {
        Self {
            weights: Array2::random((inputs, outputs), Uniform::new(-0.5, 0.5).unwrap()),
            biases: Array1::zeros(outputs),
            activation,
            last_input: None,
            last_z: None,
            accumulated_dw: None,
            accumulated_db: None,
            batch_count: 0,
        }
    }

    pub fn forward(&mut self, input: &Array1<f32>) -> Array1<f32> {
        self.last_input = Some(input.clone());

        let z = input.dot(&self.weights) + &self.biases;
        self.last_z = Some(z.clone());

        match self.activation {
            ActivationType::ReLU => z.mapv(|x| relu(x)),
            ActivationType::Sigmoid => z.mapv(|x| sigmoid(x)),
            ActivationType::SoftPlus => z.mapv(|x| softplus(x)),
            ActivationType::Tanh => z.mapv(|x| tanh(x)),
            ActivationType::Linear => z.mapv(|x| linear(x)),
        }
    }

    pub fn backward(&mut self, output_grad: Array1<f32>, _learning_rate: f32) -> Array1<f32> {
        let z = self.last_z.as_ref().unwrap();

        let grad_fn = match self.activation {
            ActivationType::ReLU => z.mapv(|x| relu_grad(x)),
            ActivationType::Sigmoid => z.mapv(|x| sigmoid_grad(x)),
            ActivationType::SoftPlus => z.mapv(|x| softplus_grad(x)),
            ActivationType::Tanh => z.mapv(|x| tanh_grad(x)),
            ActivationType::Linear => z.mapv(|x| linear_grad(x)),
        };

        let dz = output_grad * grad_fn;

        let input = self.last_input.as_ref().unwrap();
        let input_col = input.view().insert_axis(Axis(1));
        let dz_row = dz.view().insert_axis(Axis(0));
        let dw = input_col.dot(&dz_row);

        let da_prev = dz.dot(&self.weights.t());

        // Accumulate gradients
        if self.accumulated_dw.is_none() {
            self.accumulated_dw = Some(dw);
            self.accumulated_db = Some(dz);
            self.batch_count = 1;
        } else {
            if let Some(acc_dw) = &mut self.accumulated_dw {
                *acc_dw += &dw;
            }
            if let Some(acc_db) = &mut self.accumulated_db {
                *acc_db += &dz;
            }
            self.batch_count += 1;
        }

        da_prev
    }

    pub fn apply_gradients(&mut self, learning_rate: f32) {
        if self.batch_count > 0 {
            if let (Some(dw), Some(db)) = (&self.accumulated_dw, &self.accumulated_db) {
                // Average gradients over batch size
                let scale = 1.0 / self.batch_count as f32;
                self.weights = &self.weights - &(learning_rate * scale * dw);
                self.biases = &self.biases - &(learning_rate * scale * db);
            }

            // Reset accumulation
            self.accumulated_dw = None;
            self.accumulated_db = None;
            self.batch_count = 0;
        }
    }
}
