use clap::Parser;
use macroquad::prelude::*;
use rust_nn::gui::{layout, renderer::Renderer, theme};
use rust_nn::mnist::parser::MnistDataset;
use rust_nn::nn::cost::Cost;
use rust_nn::nn::layer::{ActivationType, Layer};
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
}

#[macroquad::main("RustNN")]
async fn main() {
    let args = Args::parse();

    let dataset = MnistDataset::load("mnist-dataset");
    let network = Network::new(Cost::CCE);

    request_new_screen_size(args.screen_width, args.screen_height);
    let font = load_ttf_font(theme::FONT_PATH).await.unwrap();
    let renderer = Renderer::new(font);

    let mut layout = layout::calculate_layout(screen_width(), screen_height(), &args.topology);

    loop {
        layout = layout::calculate_layout(screen_width(), screen_height(), &args.topology);

        renderer.draw_frame(&layout);

        next_frame().await;
    }
}
