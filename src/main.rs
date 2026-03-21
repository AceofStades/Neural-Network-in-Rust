use clap::Parser;
use macroquad::prelude::*;
use rust_nn::gui::bottom_panel::BottomPanel;
use rust_nn::gui::controls::ControlMessage;
use rust_nn::gui::layout_manager::LayoutManager;
use rust_nn::gui::left_panel::LeftPanel;
use rust_nn::gui::renderer::{Renderer, TrainingStats, TrainingUpdate, VisualizationData};
use rust_nn::gui::{layout, theme};
use rust_nn::mnist::parser::MnistDataset;
use rust_nn::nn::cost::Cost;
use rust_nn::nn::layer::{ActivationType, Layer};
use rust_nn::nn::network::Network;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

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
    update_tx: mpsc::Sender<TrainingUpdate>,
    control_rx: mpsc::Receiver<ControlMessage>,
    dataset: MnistDataset,
    topology: Vec<usize>,
    initial_learning_rate: f32,
    batch_size: usize,
    epochs: usize,
    model_path: Option<String>,
) {
    let mut network = Network::new(Cost::CCE);
    let mut learning_rate = initial_learning_rate;
    let mut paused = false;
    let mut speed_delay_ms = 0.0;

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
            // Check for control messages
            while let Ok(msg) = control_rx.try_recv() {
                match msg {
                    ControlMessage::Pause => {
                        paused = true;
                        println!("[Training Thread] Paused");
                    }
                    ControlMessage::Resume => {
                        paused = false;
                        println!("[Training Thread] Resumed");
                    }
                    ControlMessage::Reset => {
                        println!("[Training Thread] Resetting network...");
                        network = Network::new(Cost::CCE);
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
                        training_state.reset_epoch();
                        println!("[Training Thread] Network reset complete");
                    }
                    ControlMessage::SetLearningRate(new_lr) => {
                        learning_rate = new_lr;
                        println!("[Training Thread] Learning rate updated to {:.4}", learning_rate);
                    }
                    ControlMessage::SetSpeed(delay) => {
                        speed_delay_ms = delay;
                    }
                }
            }

            // Wait while paused
            while paused {
                thread::sleep(Duration::from_millis(100));
                // Keep checking for resume/reset messages
                if let Ok(msg) = control_rx.try_recv() {
                    match msg {
                        ControlMessage::Resume => {
                            paused = false;
                            println!("[Training Thread] Resumed");
                        }
                        ControlMessage::Reset => {
                            println!("[Training Thread] Resetting network...");
                            network = Network::new(Cost::CCE);
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
                            training_state.reset_epoch();
                            paused = false;
                            println!("[Training Thread] Network reset complete");
                            break;
                        }
                        ControlMessage::SetLearningRate(new_lr) => {
                            learning_rate = new_lr;
                        }
                        ControlMessage::SetSpeed(delay) => {
                            speed_delay_ms = delay;
                        }
                        _ => {}
                    }
                }
            }

            // Speed control delay
            if speed_delay_ms > 0.0 {
                thread::sleep(Duration::from_millis(speed_delay_ms as u64));
            }

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

            let _ = update_tx.send(update);
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

    let (update_tx, update_rx) = mpsc::channel();
    let (control_tx, control_rx) = mpsc::channel();

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
            update_tx,
            control_rx,
            dataset_clone,
            topology,
            learning_rate,
            batch_size,
            epochs,
            model_path,
        );
    });

    println!("\n[Main Thread] Rendering started. Training running in background...\n");

    let mut layout_manager = LayoutManager::new(args.screen_width, args.screen_height);
    let mut left_panel = LeftPanel::new(args.learning_rate);
    let bottom_panel = BottomPanel::new();
    let mut is_paused = false;
    let mut last_update: Option<TrainingUpdate> = None;
    let mut training_complete = false;
    let mut last_frame_time = get_time();

    loop {
        let current_time = get_time();
        let delta_time = (current_time - last_frame_time) as f32;
        last_frame_time = current_time;

        // Update layout manager with delta time for animations
        layout_manager.update(delta_time);

        // Clear background
        clear_background(theme::BACKGROUND_COLOR);

        // Calculate network layout based on available space
        let network_area = layout_manager.dimensions.network_area();
        let network_layout = layout::calculate_layout(network_area, &args.topology);

        // Handle toggle button
        if left_panel.toggle_button.update() {
            layout_manager.toggle_left_panel();
        }

        // Handle control messages from left panel
        let control_messages = left_panel.update(&layout_manager, is_paused);
        for msg in control_messages {
            if matches!(msg, ControlMessage::Pause) {
                is_paused = true;
            } else if matches!(msg, ControlMessage::Resume) {
                is_paused = false;
            }
            let _ = control_tx.send(msg);
        }

        if let Ok(update) = update_rx.try_recv() {
            last_update = Some(update.clone());
            if update.stats.epoch >= args.epochs {
                training_complete = true;
            }
        }

        // Draw network visualization
        if let Some(update) = &last_update {
            renderer.draw_frame(
                &network_layout,
                Some(&update.stats),
                Some(&update.visualization),
                layout_manager.dimensions.top_bar_height,
            );
        } else {
            renderer.draw_frame(
                &network_layout,
                None,
                None,
                layout_manager.dimensions.top_bar_height,
            );
        }

        // Draw UI panels
        left_panel.draw(&layout_manager, renderer.font());
        
        if let (Some(stats), Some(viz)) = (&last_update.as_ref().map(|u| &u.stats), &last_update.as_ref().map(|u| &u.visualization)) {
            bottom_panel.draw(&layout_manager, renderer.font(), Some(stats), Some(viz));
        } else {
            bottom_panel.draw(&layout_manager, renderer.font(), None, None);
        }

        next_frame().await;

        if training_complete && is_key_pressed(KeyCode::Escape) {
            println!("\n[Main Thread] Escape pressed. Exiting...");
            break;
        }
    }

    let _ = training_thread_handle.join();
}
