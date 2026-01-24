use macroquad::prelude::*;

pub struct Painter {
    font: Font,
}

pub fn node_size(num: u8) -> f32 {
    // FORMULA
    // (num * x) + (num - 1) * (50% of x) = 70% of screen_height()
    // (num * x) + (num - 1) * (1/2 * x) = 0.7 * screen_height()
    // x * (num + 1/2 * (num - 1)) = 0.7 * screen_height()
    // x * (2 * num + (num - 1)) / 2 = 0.7 * screen_height()
    // x = (0.7 * screen_height() * 2) / (3 * num - 1)

    (0.7 * screen_height() * 2.0) / (3.0 * num as f32 - 1.0)
}

pub fn node_pos_height(node_size: f32, node_num: u8) -> Vec<f32> {
    let start_pos: f32 = (screen_height() * 15.00 / 100.00) + node_size / 2.00;
    let mut node_pos = Vec::new();
    node_pos.push(start_pos);

    for _ in 0..node_num - 1 {
        let next_pos = node_pos[node_pos.len() - 1] + 1.5 * node_size;
        node_pos.push(next_pos);
    }
    node_pos
}

impl Painter {
    pub fn new(font: Font) -> Self {
        Self { font }
    }

    pub fn text(&self, content: &str, x: f32, y: f32, size: u16, color: Color) {
        draw_text_ex(
            content,
            x,
            y,
            TextParams {
                font: Some(&self.font),
                font_size: size,
                color: color,
                ..Default::default()
            },
        );
    }

    pub fn display(&self, num: u8) {
        clear_background(DARKGRAY);
        self.text("Neural Network Simulation", 20.0, 20.0, 30, WHITE);
        let node_size: f32 = node_size(num);
        let curr_layer_node_pos_height: Vec<f32> = node_pos_height(node_size, num);

        for i in curr_layer_node_pos_height {
            draw_circle(200.0, i, node_size / 2.0, PINK);
        }
    }
}
