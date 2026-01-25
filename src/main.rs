use clap::Parser;
use macroquad::prelude::*;
use rust_nn::gui::layout::*;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 4)]
    neurons: u8,

    #[arg(short = 'l', long = "learning-rate", default_value_t = 0.01)]
    learning_rate: f32,

    #[arg(short, long, default_value_t = String::new())]
    path: String,

    #[arg(short, long)]
    verbose: bool,

    #[arg(short = 'w', long = "screen-width", default_value_t = 1440.0)]
    screen_width: f32,

    #[arg(short = 'e', long = "screen-height", default_value_t = 900.0)]
    screen_height: f32,
}

#[macroquad::main("MyGame")]
async fn main() {
    let args = Args::parse();
    // let num: u8 = 4;
    request_new_screen_size(args.screen_width, args.screen_height);
    let font = load_ttf_font("assets/DepartureMono-Regular.otf")
        .await
        .unwrap();
    let gui: Painter = Painter::new(font);
    loop {
        gui.display(args.neurons);
        next_frame().await
    }
}
