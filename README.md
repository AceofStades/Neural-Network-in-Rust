# RustNN

![Rust](https://img.shields.io/badge/rust-v1.75%2B-orange)
![License](https://img.shields.io/badge/license-MIT-blue)
![Status](https://img.shields.io/badge/status-active-success)

RustNN is a lightweight, educational neural network library built entirely from scratch in Rust. It features a custom implementation of forward and backward propagation, varied activation functions, and a real-time visualization engine powered by Macroquad.

No PyTorch. No TensorFlow. Just pure math and Rust.

## Features

* **From-Scratch Architecture:** Complete implementation of fully connected layers, weights, and biases using `ndarray`.
* **Modular Activations:** Supports ReLU, Sigmoid, Tanh, and Linear activations.
* **Robust Cost Functions:** Includes MSE, MAE, Huber Loss, and Cross-Entropy.
* **Real-Time Visualization:** Interactive GUI to visualize network topology.
* **CLI Configurable:** Adjust hyperparameters and UI settings instantly via command-line arguments.

## Getting Started

### Prerequisites

You need the Rust toolchain installed. If you do not have it, run:

```bash
curl --proto '=https' --tlsv1.2 -sSf [https://sh.rustup.rs](https://sh.rustup.rs) | sh
```

### Installation

Clone the repository:

```bash
git clone [https://github.com/yourusername/rust-nn.git](https://github.com/yourusername/rust-nn.git)
cd rust-nn

```

## Usage (Visualizer)

You can run the neural network visualizer directly from the terminal. Use `clap` arguments to configure the simulation.

**Basic Run (Default Settings):**

```bash
cargo run
```

**Custom Configuration:**

Run with 10 hidden neurons, verbose logging, and a custom window size:

```bash
cargo run -- -n 10 -v --screen-width 1200 --screen-height 800
```

### CLI Arguments

| Flag | Long Flag | Description | Default |
| --- | --- | --- | --- |
| `-n` | `--neurons` | Number of neurons in the hidden layer | `4` |
| `-l` | `--learning-rate` | Learning rate for training | `0.01` |
| `-w` | `--screen-width` | Window width in pixels | `1440.0` |
| `-h` | `--screen-height` | Window height in pixels | `900.0` |
| `-v` | `--verbose` | Enable verbose logging | `false` |
|  | `--path` | Path to load/save model | *Required* |

## Library Usage (The Math)

You can also use the internal `nn` module to build networks programmatically for solving logic gates (XOR) or regression problems.

```rust
use rust_nn::nn::{Network, Layer, Cost, ActivationType};
use ndarray::arr1;

fn main() {
    // 1. Create a Network
    let mut net = Network::new(Cost::MSE);

    // 2. Add Layers
    net.add(Layer::new(2, 4, ActivationType::Tanh));    // Input -> Hidden
    net.add(Layer::new(4, 1, ActivationType::Sigmoid)); // Hidden -> Output

    // 3. Train (XOR Example)
    let inputs = vec![arr1(&[0.0, 0.0]), arr1(&[1.0, 0.0])];
    let targets = vec![arr1(&[0.0]), arr1(&[1.0])];

    net.train(&inputs, &targets, 10_000, 0.2);
}

```

## Project Structure

```text
rust-nn/
├── src/
│   ├── main.rs           # Entry point (CLI & Visualizer Loop)
│   ├── lib.rs            # Library exports
│   │
│   ├── nn/               # Core Logic
│   │   ├── network.rs    # Network struct & training loop
│   │   ├── layer.rs      # Layer struct & backprop logic
│   │   ├── cost.rs       # Loss functions (MSE, CCE, etc.)
│   │   └── activation.rs # ReLU, Sigmoid, Tanh, Linear
│   │
│   └── gui/              # Visualization
│       ├── mod.rs
│       └── layout.rs     # Macroquad drawing logic
│
└── tests/                # Integration tests (XOR, Linear Regression)

```

## Running Tests

Ensure the math is solid by running the integration tests. This verifies that the network can actually learn XOR and Linear Regression tasks.

```bash
cargo test

```

## Tech Stack

* **Language:** Rust
* **Math:** `ndarray` (Linear Algebra), `rand` (Initialization)
* **GUI:** `macroquad` (Immediate Mode Graphics)
* **CLI:** `clap` (Argument Parsing)

## Roadmap

* [ ] Add multithreading to separate Training from Rendering.
* [ ] Implement dataset loading (MNIST).
* [ ] Add interactive UI controls (Buttons/Sliders).
* [ ] Save/Load trained models to disk.

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
