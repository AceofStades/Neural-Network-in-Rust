use super::layout::NetworkLayout;
use super::theme::*;
use macroquad::prelude::*;

pub struct Renderer {
    font: Font,
}

impl Renderer {
    pub fn new(font: Font) -> Self {
        Self { font }
    }

    pub fn draw_frame(&self, layout: &NetworkLayout) {
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
}
