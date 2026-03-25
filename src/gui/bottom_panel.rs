use super::layout_manager::{LayoutManager, PANEL_PADDING};
use super::renderer::{TrainingStats, VisualizationData};
use macroquad::prelude::*;

pub struct BottomPanel;

impl BottomPanel {
    pub fn new() -> Self {
        Self
    }

    pub fn draw(
        &self,
        layout: &LayoutManager,
        font: &Font,
        stats: Option<&TrainingStats>,
        viz: Option<&VisualizationData>,
    ) {
        let bounds = layout.dimensions.bottom_panel_bounds();

        // Draw shadow for depth
        draw_rectangle(
            bounds.x,
            bounds.y - 3.0,
            bounds.width,
            3.0,
            Color::new(0.0, 0.0, 0.0, 0.3),
        );

        // Draw panel background
        draw_rectangle(
            bounds.x,
            bounds.y,
            bounds.width,
            bounds.height,
            Color::new(0.12, 0.12, 0.12, 0.95),
        );
        
        // Draw border
        draw_rectangle_lines(
            bounds.x,
            bounds.y,
            bounds.width,
            bounds.height,
            2.0,
            Color::new(0.35, 0.35, 0.35, 1.0),
        );

        if let Some(s) = stats {
            self.draw_stats(bounds.x, bounds.y, bounds.width, font, s);
        }

        if let Some(v) = viz {
            self.draw_progress_bar(bounds.x, bounds.y, bounds.width, bounds.height, font, v);
        }
    }

    fn draw_stats(&self, x: f32, y: f32, width: f32, font: &Font, stats: &TrainingStats) {
        let stats_y = y + PANEL_PADDING;
        let stats_x = x + PANEL_PADDING;

        // Compact single-line layout for stats
        let epoch_text = format!("Epoch: {}", stats.epoch);
        let loss_text = format!("Loss: {:.4}", stats.loss);
        let acc_text = format!("Accuracy: {:.2}%", stats.accuracy * 100.0);
        let val_acc_text = if let Some(val_acc) = stats.val_accuracy {
            format!("Val Acc: {:.2}%", val_acc * 100.0)
        } else {
            "Val Acc: --".to_string()
        };

        // Calculate spacing
        let total_width = width - PANEL_PADDING * 2.0;
        let spacing = total_width / 4.0;

        draw_text_ex(
            &epoch_text,
            stats_x,
            stats_y + 15.0,
            TextParams {
                font: Some(font),
                font_size: 18,
                color: WHITE,
                ..Default::default()
            },
        );

        draw_text_ex(
            &loss_text,
            stats_x + spacing,
            stats_y + 15.0,
            TextParams {
                font: Some(font),
                font_size: 18,
                color: WHITE,
                ..Default::default()
            },
        );

        draw_text_ex(
            &acc_text,
            stats_x + spacing * 2.0,
            stats_y + 15.0,
            TextParams {
                font: Some(font),
                font_size: 18,
                color: WHITE,
                ..Default::default()
            },
        );

        draw_text_ex(
            &val_acc_text,
            stats_x + spacing * 3.0,
            stats_y + 15.0,
            TextParams {
                font: Some(font),
                font_size: 18,
                color: Color::new(0.4, 0.8, 1.0, 1.0), // Light blue for validation
                ..Default::default()
            },
        );
    }

    fn draw_progress_bar(
        &self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        font: &Font,
        viz: &VisualizationData,
    ) {
        let bar_width = width - PANEL_PADDING * 4.0 - 80.0; // Leave space for percentage text
        let bar_height = 20.0;
        let bar_x = x + PANEL_PADDING * 2.0;
        let bar_y = y + height - bar_height - PANEL_PADDING;

        // Background
        draw_rectangle(
            bar_x,
            bar_y,
            bar_width,
            bar_height,
            Color::new(0.3, 0.3, 0.3, 1.0),
        );

        // Progress fill
        draw_rectangle(
            bar_x,
            bar_y,
            bar_width * viz.epoch_progress.min(1.0),
            bar_height,
            Color::new(0.2, 0.8, 0.3, 1.0),
        );

        // Border
        draw_rectangle_lines(bar_x, bar_y, bar_width, bar_height, 2.0, WHITE);

        // Percentage text
        let progress_text = format!("{:.0}%", viz.epoch_progress * 100.0);
        let text_dims = measure_text(&progress_text, Some(font), 16, 1.0);
        let text_x = bar_x + bar_width + 10.0;
        let text_y = bar_y + (bar_height + text_dims.height) / 2.0 - 2.0;

        draw_text_ex(
            &progress_text,
            text_x,
            text_y,
            TextParams {
                font: Some(font),
                font_size: 16,
                color: WHITE,
                ..Default::default()
            },
        );
    }
}
