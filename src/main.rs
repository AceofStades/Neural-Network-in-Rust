use macroquad::prelude::*;

fn node_size(num: u8, screen_height: f32) -> f32 {
    (screen_height / 2.0) / num as f32
}

// FORMULA
// (num * x) + num * (50% of x) = 70% of screen_height()
// x = (2 * 70 * screen_height()) / (3 * num_nodes_in_a_layer)

#[macroquad::main("MyGame")]
async fn main() {
    let num: u8 = 5;
    loop {
        clear_background(DARKGRAY);
        request_new_screen_size(1440.0, 900.0);
        draw_text("Neural Network Simulation", 20.0, 20.0, 30.0, WHITE);
        let node_size: f32 = node_size(num, screen_height());

        for _ in 0..num {
            draw_circle(x, y, r, color);
        }

        next_frame().await
    }
}
