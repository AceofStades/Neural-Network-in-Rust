use macroquad::prelude::*;
use rust_nn::gui::layout::*;

#[macroquad::main("MyGame")]
async fn main() {
    let num: u8 = 4;
    request_new_screen_size(1440.0, 900.0);
    let font = load_ttf_font("assets/DepartureMono-Regular.otf")
        .await
        .unwrap();
    let gui: Painter = Painter::new(font);
    loop {
        gui.display(num);
        next_frame().await
    }
}
