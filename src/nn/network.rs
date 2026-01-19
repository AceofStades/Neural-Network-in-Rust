use ndarray::Array1;

use crate::nn::layer::Layer;

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    pub fn predict(&mut self, input: Array1<f32>) -> Array1<f32> {
        let mut output = input;

        for layer in &mut self.layers {
            output = layer.forward(&output);
        }

        output
    }

    pub fn train_one_epoch(
        &mut self,
        inputs: &Vec<Array1<f32>>,
        targets: &Vec<Array1<f32>>,
        learning_rate: f32,
    ) {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let prediction = self.predict(input.clone());
            let mut gradient = &prediction - target;

            for layer in self.layers.iter_mut().rev() {
                gradient = layer.backward(gradient, learning_rate);
            }
        }
    }
}
