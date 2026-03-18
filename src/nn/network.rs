use ndarray::Array1;

use crate::nn::cost::Cost;
use crate::nn::layer::Layer;

#[derive(Clone)]
pub struct Network {
    pub layers: Vec<Layer>,
    pub cost: Cost,
    pub activations: Vec<Array1<f32>>,
}

impl Network {
    pub fn new(cost: Cost) -> Self {
        Self {
            layers: Vec::new(),
            cost,
            activations: Vec::new(),
        }
    }

    pub fn add(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    pub fn predict(&mut self, input: Array1<f32>) -> Array1<f32> {
        let mut output = input.clone();
        self.activations.clear();
        self.activations.push(input);

        for layer in &mut self.layers {
            output = layer.forward(&output);
            self.activations.push(output.clone());
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
            let mut gradient = self.cost.prime(target, &prediction);

            for layer in self.layers.iter_mut().rev() {
                gradient = layer.backward(gradient, learning_rate);
            }
        }
    }
}
