use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::fs::File;

use crate::nn::cost::Cost;
use crate::nn::layer::Layer;

#[derive(Clone, Serialize, Deserialize)]
pub struct Network {
    pub layers: Vec<Layer>,
    pub cost: Cost,
    #[serde(skip)]
    pub activations: Vec<Array1<f32>>,
}

impl Network {
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        serde_json::to_writer(file, self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let network: Network = serde_json::from_reader(file)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(network)
    }

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

    pub fn train_batch(
        &mut self,
        inputs: &[Array1<f32>],
        targets: &[Array1<f32>],
        learning_rate: f32,
    ) {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let prediction = self.predict(input.clone());
            let mut gradient = self.cost.prime(target, &prediction);

            for layer in self.layers.iter_mut().rev() {
                gradient = layer.backward(gradient, learning_rate);
            }
        }
        
        // Apply accumulated gradients after processing the batch
        for layer in &mut self.layers {
            layer.apply_gradients(learning_rate);
        }
    }

    pub fn evaluate(
        &mut self,
        inputs: &[Array1<f32>],
        targets: &[Array1<f32>],
    ) -> (f32, f32) {
        let mut total_loss = 0.0;
        let mut total_correct = 0;
        let mut samples = 0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let prediction = self.predict(input.clone());
            
            // Calculate loss (using existing cost function)
            let loss = self.cost.calc(target, &prediction);
            total_loss += loss;

            let pred_idx = prediction
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            
            let target_idx = target
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            if pred_idx == target_idx {
                total_correct += 1;
            }
            samples += 1;
        }

        let avg_loss = if samples > 0 { total_loss / samples as f32 } else { 0.0 };
        let accuracy = if samples > 0 { total_correct as f32 / samples as f32 } else { 0.0 };

        (avg_loss, accuracy)
    }
}
