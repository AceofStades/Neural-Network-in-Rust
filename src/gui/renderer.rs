use super::layout::NetworkLayout;
use super::theme::*;
use macroquad::prelude::*;
use ndarray::Array1;

#[derive(Clone)]
pub struct TrainingStats {
    pub epoch: usize,
    pub loss: f32,
    pub accuracy: f32,
    pub batch_count: usize,
}

#[derive(Clone)]
pub struct VisualizationData {
    pub activations: Vec<Array1<f32>>,
    pub prediction: Option<usize>,
    pub target: Option<usize>,
    pub confidence: f32,
    pub epoch_progress: f32,
}

#[derive(Clone)]
pub struct TrainingUpdate {
    pub stats: TrainingStats,
    pub visualization: VisualizationData,
}

pub struct Renderer {
    font: Font,
}

impl Renderer {
    pub fn new(font: Font) -> Self {
        Self { font }
    }

    pub fn font(&self) -> &Font {
        &self.font
    }

    pub fn draw_frame(
        &self,
        layout: &NetworkLayout,
        _stats: Option<&TrainingStats>,
        viz: Option<&VisualizationData>,
        title_y: f32,
    ) {
        self.draw_text_centered("Neural Network Simulation", title_y + 30.0, 30);

        // Draw connections
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

        // Draw neurons with activation-based coloring
        if let Some(v) = viz {
            self.draw_neurons_with_activations(layout, &v.activations);
        } else {
            // Static rendering if no visualization data
            for layer in &layout.node_positions {
                for pos in layer {
                    draw_circle(pos.x, pos.y, layout.node_radius, NODE_COLOR);
                    draw_circle_lines(pos.x, pos.y, layout.node_radius, 2.0, WHITE);
                }
            }
        }

        // Draw layer node counts
        self.draw_layer_counts(layout);

        // Draw prediction info (top right)
        if let Some(v) = viz {
            self.draw_prediction_info(v);
        }
    }

    fn draw_neurons_with_activations(&self, layout: &NetworkLayout, activations: &[Array1<f32>]) {
        for (layer_idx, layer_positions) in layout.node_positions.iter().enumerate() {
            if layer_idx < activations.len() && layer_idx < layout.node_display_info.len() {
                let layer_activations = &activations[layer_idx];
                let layer_info = &layout.node_display_info[layer_idx];

                for (pos_idx, pos) in layer_positions.iter().enumerate() {
                    if let Some(info) = layer_info.get(pos_idx) {
                        if info.is_real {
                            let activation = if info.index < layer_activations.len() {
                                layer_activations[info.index].max(0.0).min(1.0)
                            } else {
                                0.0
                            };

                            let color = self.activation_to_color(activation);
                            draw_circle(pos.x, pos.y, layout.node_radius, color);
                            draw_circle_lines(pos.x, pos.y, layout.node_radius, 2.0, WHITE);
                        } else {
                            let dot_size = layout.node_radius * 0.3;
                            draw_circle(pos.x - dot_size - 2.0, pos.y, dot_size, Color::new(0.6, 0.6, 0.6, 0.7));
                            draw_circle(pos.x, pos.y, dot_size, Color::new(0.6, 0.6, 0.6, 0.7));
                            draw_circle(pos.x + dot_size + 2.0, pos.y, dot_size, Color::new(0.6, 0.6, 0.6, 0.7));
                        }
                    }
                }
            }
        }
    }

    fn activation_to_color(&self, activation: f32) -> Color {
        if activation < 0.25 {
            Color::new(0.2, 0.2, 0.2, 1.0)
        } else if activation < 0.5 {
            Color::new(0.4, 0.4, 0.6, 1.0)
        } else if activation < 0.75 {
            Color::new(0.6, 0.8, 0.3, 1.0)
        } else {
            Color::new(1.0, 0.8, 0.0, 1.0)
        }
    }

    fn draw_prediction_info(&self, viz: &VisualizationData) {
        let x = screen_width() - 250.0;
        let y = 80.0;

        if let Some(pred) = viz.prediction {
            self.draw_text_left(&format!("Prediction: {}", pred), x, y, 24);
        }

        if let Some(target) = viz.target {
            self.draw_text_left(&format!("Target: {}", target), x, y + 30.0, 24);
        }

        self.draw_text_left(
            &format!("Confidence: {:.1}%", viz.confidence * 100.0),
            x,
            y + 60.0,
            20,
        );
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

    fn draw_layer_counts(&self, layout: &NetworkLayout) {
        for (layer_idx, layer_positions) in layout.node_positions.iter().enumerate() {
            if let Some(first_pos) = layer_positions.first() {
                let node_count = layout.layer_sizes[layer_idx];
                let count_text = format!("{}", node_count);
                let text_dims = measure_text(&count_text, Some(&self.font), 18, 1.0);
                let text_x = first_pos.x - (text_dims.width / 2.0);
                let text_y = first_pos.y - layout.node_radius - 15.0;
                
                self.draw_text_left(&count_text, text_x, text_y, 18);
            }
        }
    }
}
