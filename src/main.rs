use clap::Parser;
use macroquad::prelude::*;
use rust_nn::gui::renderer::{Renderer, TrainingStats, TrainingUpdate, VisualizationData};
use rust_nn::gui::{layout, theme};
use rust_nn::mnist::parser::MnistDataset;
use rust_nn::nn::cost::Cost;
use rust_nn::nn::layer::{ActivationType, Layer};
use rust_nn::nn::network::Network;
use std::sync::mpsc;
use std::thread;

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

    #[arg(short = 'p', long)]
    path: Option<String>,
}

struct TrainingState {
    batch_index: usize,
    total_loss: f32,
    total_correct: usize,
    total_samples: usize,
}

impl TrainingState {
    fn new() -> Self {
        Self {
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

fn training_thread(
    tx: mpsc::Sender<TrainingUpdate>,
    dataset: MnistDataset,
    topology: Vec<usize>,
    learning_rate: f32,
    batch_size: usize,
    epochs: usize,
    model_path: Option<String>,
) {
    let mut network = Network::new(Cost::CCE);

    for i in 0..(topology.len() - 1) {
        let input_size = topology[i];
        let output_size = topology[i + 1];

        let activation = if i == topology.len() - 2 {
            ActivationType::Sigmoid
        } else {
            ActivationType::Tanh
        };

        network.add(Layer::new(input_size, output_size, activation));
    }

    if let Some(path) = &model_path {
        if std::path::Path::new(path).exists() {
            println!("[Training Thread] Loading model from {}", path);
            match Network::load(path) {
                Ok(loaded_net) => {
                    network = loaded_net;
                }
                Err(e) => {
                    println!(
                        "[Training Thread] Failed to load model: {}. Using new network.",
                        e
                    );
                }
            }
        }
    }

    println!(
        "[Training Thread] Network initialized: {} layers",
        network.layers.len()
    );
    println!(
        "[Training Thread] Config: {} epochs, batch_size={}, learning_rate={}",
        epochs, batch_size, learning_rate
    );
    println!("[Training Thread] Starting training...\n");

    let mut training_state = TrainingState::new();
    let total_train_samples = dataset.train_images.len();

    for epoch in 0..epochs {
        training_state.reset_epoch();

        for batch_idx in 0..((total_train_samples + batch_size - 1) / batch_size) {
            let batch_start = batch_idx * batch_size;
            let batch_end = (batch_start + batch_size).min(total_train_samples);

            for idx in batch_start..batch_end {
                let prediction = network.predict(dataset.train_images[idx].clone());
                let target = &dataset.train_labels[idx];

                let loss: f32 = (prediction.iter().zip(target.iter()))
                    .map(|(p, t)| (p - t).powi(2))
                    .sum();
                training_state.total_loss += loss;

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

                network.train_one_epoch(
                    &vec![dataset.train_images[idx].clone()],
                    &vec![dataset.train_labels[idx].clone()],
                    learning_rate,
                );
            }

            training_state.batch_index += 1;

            let epoch_progress = (batch_start as f32 + (batch_end - batch_start) as f32)
                / total_train_samples as f32;

            let update = TrainingUpdate {
                stats: TrainingStats {
                    epoch: epoch + 1,
                    loss: training_state.get_loss(),
                    accuracy: training_state.get_accuracy(),
                    batch_count: training_state.batch_index,
                },
                visualization: VisualizationData {
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
                },
            };

            let _ = tx.send(update);
        }

        println!(
            "Epoch {}/{} | Loss: {:.4} | Accuracy: {:.2}%",
            epoch + 1,
            epochs,
            training_state.get_loss(),
            training_state.get_accuracy() * 100.0
        );
    }

    println!(
        "\n[Training Thread] Training complete! Final accuracy: {:.2}%",
        training_state.get_accuracy() * 100.0
    );

    if let Some(path) = &model_path {
        println!("[Training Thread] Saving model to {}", path);
        if let Err(e) = network.save(path) {
            println!("[Training Thread] Failed to save model: {}", e);
        }
    }
}

#[macroquad::main("RustNN")]
async fn main() {
    let args = Args::parse();

    let dataset = MnistDataset::load("mnist-dataset");

    request_new_screen_size(args.screen_width, args.screen_height);
    let font = load_ttf_font(theme::FONT_PATH).await.unwrap();
    let renderer = Renderer::new(font);

    let (tx, rx) = mpsc::channel();

    let topology = args.topology.clone();
    let dataset_clone = MnistDataset {
        train_images: dataset.train_images.clone(),
        train_labels: dataset.train_labels.clone(),
        test_images: dataset.test_images.clone(),
        test_labels: dataset.test_labels.clone(),
    };

    let learning_rate = args.learning_rate;
    let batch_size = args.batch_size;
    let epochs = args.epochs;
    let model_path = args.path.clone();

    let training_thread_handle = thread::spawn(move || {
        training_thread(
            tx,
            dataset_clone,
            topology,
            learning_rate,
            batch_size,
            epochs,
            model_path,
        );
    });

    println!("\n[Main Thread] Rendering started. Training running in background...\n");

    let mut last_update: Option<TrainingUpdate> = None;
    let mut training_complete = false;

    loop {
        let layout = layout::calculate_layout(screen_width(), screen_height(), &args.topology);

        if let Ok(update) = rx.try_recv() {
            last_update = Some(update.clone());
            if update.stats.epoch >= args.epochs {
                training_complete = true;
            }
        }

        if let Some(update) = &last_update {
            renderer.draw_frame(&layout, Some(&update.stats), Some(&update.visualization));
        } else {
            renderer.draw_frame(&layout, None, None);
        }

        next_frame().await;

        if training_complete && is_key_pressed(KeyCode::Escape) {
            println!("\n[Main Thread] Escape pressed. Exiting...");
            break;
        }
    }

    let _ = training_thread_handle.join();
}
