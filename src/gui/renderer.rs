use super::layout::NetworkLayout;
use super::theme::*;
use macroquad::prelude::*;

pub struct TrainingStats {
    pub epoch: usize,
    pub loss: f32,
    pub accuracy: f32,
    pub batch_count: usize,
}

pub struct Renderer {
    font: Font,
}

impl Renderer {
    pub fn new(font: Font) -> Self {
        Self { font }
    }

    pub fn draw_frame(&self, layout: &NetworkLayout, stats: Option<&TrainingStats>) {
        clear_background(BACKGROUND_COLOR);

        self.draw_text_centered("Neural Network Simulation", 50.0, 30);

        for (start, end) in &layout.connections {
            draw_line(
                start.x + 5.0,
                start.y,
                end.x - 5.0,
                end.y,
                2.0,
                Color::new(0.8, 0.8, 0.8, 0.5),
            );
        }
        for layer in &layout.node_positions {
            for pos in layer {
                draw_circle(pos.x, pos.y, layout.node_radius, NODE_COLOR);
                draw_circle_lines(pos.x, pos.y, layout.node_radius, 2.0, WHITE);
            }
        }

        if let Some(s) = stats {
            self.draw_training_stats(s);
        }
    }

    fn draw_training_stats(&self, stats: &TrainingStats) {
        let padding = 20.0;
        let y_start = screen_height() - 120.0;

        let epoch_text = format!("Epoch: {}", stats.epoch);
        let loss_text = format!("Loss: {:.4}", stats.loss);
        let acc_text = format!("Accuracy: {:.2}%", stats.accuracy * 100.0);
        let batch_text = format!("Batches: {}", stats.batch_count);

        self.draw_text_left(&epoch_text, padding, y_start, 20);
        self.draw_text_left(&loss_text, padding, y_start + 25.0, 20);
        self.draw_text_left(&acc_text, padding, y_start + 50.0, 20);
        self.draw_text_left(&batch_text, padding, y_start + 75.0, 20);
    }

    fn draw_text_centered(&self, text: &str, y: f32, size: u16) {
        let dims = measure_text(text, Some(&self.font), size, 1.0);
        let x = (screen_width() - dims.width) / 2.0;

        draw_text_ex(
            text,
            x,
            y,
            TextParams {
                font: Some(&self.font),
                font_size: size,
                color: TEXT_COLOR,
                ..Default::default()
            },
        );
    }

    fn draw_text_left(&self, text: &str, x: f32, y: f32, size: u16) {
        draw_text_ex(
            text,
            x,
            y,
            TextParams {
                font: Some(&self.font),
                font_size: size,
                color: TEXT_COLOR,
                ..Default::default()
            },
        );
    }
}
