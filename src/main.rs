use clap::Parser;
use macroquad::prelude::*;
use rust_nn::gui::renderer::{Renderer, TrainingStats, VisualizationData};
use rust_nn::gui::{layout, theme};
use rust_nn::mnist::parser::MnistDataset;
use rust_nn::nn::cost::Cost;
use rust_nn::nn::layer::{Layer, ActivationType};
use rust_nn::nn::network::Network;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 't', long, value_delimiter = ' ', num_args = 1.., default_value = "784 320 100 10")]
    topology: Vec<usize>,

    #[arg(short = 'w', long, default_value_t = 1440.0)]
    screen_width: f32,

    #[arg(short = 'e', long, default_value_t = 900.0)]
    screen_height: f32,

    #[arg(short = 'l', long, default_value_t = 0.1)]
    learning_rate: f32,

    #[arg(short = 'b', long, default_value_t = 32)]
    batch_size: usize,

    #[arg(short = 'E', long, default_value_t = 5)]
    epochs: usize,
}

struct TrainingState {
    epoch: usize,
    batch_index: usize,
    total_loss: f32,
    total_correct: usize,
    total_samples: usize,
}

impl TrainingState {
    fn new() -> Self {
        Self {
            epoch: 0,
            batch_index: 0,
            total_loss: 0.0,
            total_correct: 0,
            total_samples: 0,
        }
    }

    fn reset_epoch(&mut self) {
        self.batch_index = 0;
        self.total_loss = 0.0;
        self.total_correct = 0;
        self.total_samples = 0;
    }

    fn get_loss(&self) -> f32 {
        if self.total_samples == 0 {
            0.0
        } else {
            self.total_loss / self.total_samples as f32
        }
    }

    fn get_accuracy(&self) -> f32 {
        if self.total_samples == 0 {
            0.0
        } else {
            self.total_correct as f32 / self.total_samples as f32
        }
    }
}

#[macroquad::main("RustNN")]
async fn main() {
    let args = Args::parse();

    let dataset = MnistDataset::load("mnist-dataset");
    let mut network = Network::new(Cost::CCE);

    // Build network from topology
    for i in 0..(args.topology.len() - 1) {
        let input_size = args.topology[i];
        let output_size = args.topology[i + 1];

        let activation = if i == args.topology.len() - 2 {
            ActivationType::Sigmoid
        } else {
            ActivationType::Tanh
        };

        network.add(Layer::new(input_size, output_size, activation));
    }

    request_new_screen_size(args.screen_width, args.screen_height);
    let font = load_ttf_font(theme::FONT_PATH).await.unwrap();
    let renderer = Renderer::new(font);

    println!(
        "✓ Network initialized: {} layers",
        network.layers.len()
    );
    println!(
        "📚 Training config: {} epochs, batch_size={}, learning_rate={}",
        args.epochs, args.batch_size, args.learning_rate
    );
    println!("Starting training...\n");

    let mut training_state = TrainingState::new();

    loop {
        let layout = layout::calculate_layout(screen_width(), screen_height(), &args.topology);

        // Process one batch per frame for smooth animation
        if training_state.epoch < args.epochs && training_state.batch_index * args.batch_size < dataset.train_images.len() {
            let batch_start = training_state.batch_index * args.batch_size;
            let batch_end = (batch_start + args.batch_size).min(dataset.train_images.len());

            for idx in batch_start..batch_end {
                let prediction = network.predict(dataset.train_images[idx].clone());
                let target = &dataset.train_labels[idx];

                // Calculate loss (simplified - just use first output as loss)
                let loss: f32 = (prediction.iter().zip(target.iter()))
                    .map(|(p, t)| (p - t).powi(2))
                    .sum();
                training_state.total_loss += loss;

                // Check if prediction is correct (argmax match)
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
                    training_state.total_correct += 1;
                }

                training_state.total_samples += 1;

                // Train on this sample
                network.train_one_epoch(
                    &vec![dataset.train_images[idx].clone()],
                    &vec![dataset.train_labels[idx].clone()],
                    args.learning_rate,
                );
            }

            training_state.batch_index += 1;

            // Check if epoch is complete
            if batch_end >= dataset.train_images.len() {
                let epoch_num = training_state.epoch + 1;
                let avg_loss = training_state.get_loss();
                let accuracy = training_state.get_accuracy();
                println!(
                    "Epoch {}/{} | Loss: {:.4} | Accuracy: {:.2}%",
                    epoch_num,
                    args.epochs,
                    avg_loss,
                    accuracy * 100.0
                );

                training_state.epoch += 1;
                training_state.reset_epoch();
            }
        }

        let stats = TrainingStats {
            epoch: training_state.epoch,
            loss: training_state.get_loss(),
            accuracy: training_state.get_accuracy(),
            batch_count: training_state.batch_index,
        };

        // Create visualization data from network state
        let epoch_progress = if dataset.train_images.is_empty() {
            0.0
        } else {
            let total_samples = dataset.train_images.len();
            let current_sample = (training_state.batch_index * args.batch_size).min(total_samples);
            current_sample as f32 / total_samples as f32
        };

        let viz = VisualizationData {
            activations: network.activations.clone(),
            prediction: network.activations.last().and_then(|out| {
                out.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
            }),
            target: None,
            confidence: network
                .activations
                .last()
                .map(|out| out.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
                .unwrap_or(0.0)
                .max(0.0)
                .min(1.0),
            epoch_progress,
        };

        // Check if training is complete and print summary
        if training_state.epoch >= args.epochs && training_state.batch_index > 0 {
            println!("\n✓ Training complete! Final accuracy: {:.2}%", stats.accuracy * 100.0);
            println!("Network is ready for inference. Visualizer will continue running.");
        }

        renderer.draw_frame(&layout, Some(&stats), Some(&viz));
        next_frame().await;
    }
}
