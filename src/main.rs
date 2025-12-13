use macroquad::prelude::*;

fn node_size(num: u8) -> f32 {
    // FORMULA
    // (num * x) + (num - 1) * (50% of x) = 70% of screen_height()
    // (num * x) + (num - 1) * (1/2 * x) = 0.7 * screen_height()
    // x * (num + 1/2 * (num - 1)) = 0.7 * screen_height()
    // x * (2 * num + (num - 1)) / 2 = 0.7 * screen_height()
    // x = (0.7 * screen_height() * 2) / (3 * num - 1)

    (0.7 * screen_height() * 2.0) / (3.0 * num as f32 - 1.0)
}

fn node_pos_height(node_size: f32, node_num: u8) -> Vec<f32> {
    let start_pos: f32 = (screen_height() * 15.00 / 100.00) + node_size / 2.00;
    let mut node_pos = Vec::new();
    node_pos.push(start_pos);

    for _ in 0..node_num - 1 {
        let next_pos = node_pos[node_pos.len() - 1] + 1.5 * node_size;
        node_pos.push(next_pos);
    }
    println!("Screen Height: {}", screen_height());
    println!("Node Size: {}", node_size);
    println!("Start Position: {}", start_pos);
    println!("{:?}", node_pos);
    node_pos
}

#[macroquad::main("MyGame")]
async fn main() {
    let num: u8 = 4;
    request_new_screen_size(1440.0, 900.0);
    loop {
        clear_background(DARKGRAY);
        draw_text("Neural Network Simulation", 20.0, 20.0, 30.0, WHITE);
        let node_size: f32 = node_size(num);
        let curr_layer_node_pos_height: Vec<f32> = node_pos_height(node_size, num);

        for i in curr_layer_node_pos_height {
            draw_circle(200.0, i, node_size / 2.0, PINK);
        }

        next_frame().await
    }
}
