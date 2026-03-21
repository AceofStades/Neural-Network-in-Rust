use super::controls::{Button, ControlMessage, Slider};
use super::layout_manager::{LayoutManager, PANEL_PADDING, TOGGLE_BUTTON_SIZE};
use macroquad::prelude::*;

pub struct LeftPanel {
    pub toggle_button: ToggleButton,
    pub play_pause_button: Button,
    pub reset_button: Button,
    pub learning_rate_slider: Slider,
    pub speed_slider: Slider,
}

impl LeftPanel {
    pub fn new(initial_lr: f32) -> Self {
        Self {
            toggle_button: ToggleButton::new(),
            play_pause_button: Button::new(0.0, 0.0, 0.0, 0.0, "Pause"),
            reset_button: Button::new(0.0, 0.0, 0.0, 0.0, "Reset"),
            learning_rate_slider: Slider::new(0.0, 0.0, 0.0, "Learning Rate", 0.001, 1.0, initial_lr),
            speed_slider: Slider::new(0.0, 0.0, 0.0, "Speed (ms)", 0.0, 100.0, 0.0),
        }
    }

    pub fn update(&mut self, layout: &LayoutManager, is_paused: bool) -> Vec<ControlMessage> {
        let mut messages = Vec::new();
        let bounds = layout.dimensions.left_panel_bounds();

        // Update toggle button position
        self.toggle_button.x = bounds.x + PANEL_PADDING;
        self.toggle_button.y = bounds.y + PANEL_PADDING;
        self.toggle_button.collapsed = layout.is_collapsed();

        // Only update controls if panel is expanded
        if !layout.is_collapsed() && !layout.is_animating() {
            let button_width = bounds.width - PANEL_PADDING * 2.0;
            let button_height = 40.0;
            let spacing = 15.0;

            let mut current_y = bounds.y + PANEL_PADDING + TOGGLE_BUTTON_SIZE + spacing;

            // Update play/pause button
            self.play_pause_button.x = bounds.x + PANEL_PADDING;
            self.play_pause_button.y = current_y;
            self.play_pause_button.width = button_width;
            self.play_pause_button.height = button_height;
            self.play_pause_button.label = if is_paused { "Resume" } else { "Pause" }.to_string();

            if self.play_pause_button.update() {
                messages.push(if is_paused {
                    ControlMessage::Resume
                } else {
                    ControlMessage::Pause
                });
            }

            current_y += button_height + spacing;

            // Update reset button
            self.reset_button.x = bounds.x + PANEL_PADDING;
            self.reset_button.y = current_y;
            self.reset_button.width = button_width;
            self.reset_button.height = button_height;

            if self.reset_button.update() {
                messages.push(ControlMessage::Reset);
            }

            current_y += button_height + spacing * 2.0;

            // Update learning rate slider
            self.learning_rate_slider.x = bounds.x + PANEL_PADDING;
            self.learning_rate_slider.y = current_y;
            self.learning_rate_slider.width = button_width;

            if self.learning_rate_slider.update() {
                messages.push(ControlMessage::SetLearningRate(self.learning_rate_slider.value));
            }

            current_y += 50.0;

            // Update speed slider
            self.speed_slider.x = bounds.x + PANEL_PADDING;
            self.speed_slider.y = current_y;
            self.speed_slider.width = button_width;

            if self.speed_slider.update() {
                messages.push(ControlMessage::SetSpeed(self.speed_slider.value));
            }
        }

        messages
    }

    pub fn draw(&self, layout: &LayoutManager, font: &Font) {
        let bounds = layout.dimensions.left_panel_bounds();

        // Draw shadow for depth
        draw_rectangle(
            bounds.x + 3.0,
            bounds.y + 3.0,
            bounds.width,
            bounds.height,
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
        
        // Draw border with slight gradient effect
        draw_rectangle_lines(
            bounds.x,
            bounds.y,
            bounds.width,
            bounds.height,
            2.0,
            Color::new(0.35, 0.35, 0.35, 1.0),
        );

        // Always draw toggle button
        self.toggle_button.draw(font);

        // Only draw controls if panel is expanded
        if !layout.is_collapsed() {
            self.play_pause_button.draw(font);
            self.reset_button.draw(font);
            self.learning_rate_slider.draw(font);
            self.speed_slider.draw(font);

            // Draw legend
            self.draw_legend(layout, font);
        }
    }

    fn draw_legend(&self, layout: &LayoutManager, font: &Font) {
        let bounds = layout.dimensions.left_panel_bounds();
        let legend_y = bounds.y + bounds.height - 180.0;
        let x = bounds.x + PANEL_PADDING;

        // Title
        draw_text_ex(
            "Legend",
            x,
            legend_y,
            TextParams {
                font: Some(font),
                font_size: 18,
                color: WHITE,
                ..Default::default()
            },
        );

        let item_y_start = legend_y + 25.0;
        let item_spacing = 30.0;
        let circle_radius = 8.0;
        let circle_x = x + circle_radius;

        // Color samples and labels
        let items = [
            (Color::new(0.2, 0.2, 0.2, 1.0), "Low (< 0.25)"),
            (Color::new(0.4, 0.4, 0.6, 1.0), "Medium (< 0.5)"),
            (Color::new(0.6, 0.8, 0.3, 1.0), "High (< 0.75)"),
            (Color::new(1.0, 0.8, 0.0, 1.0), "Very High (≥ 0.75)"),
        ];

        for (i, (color, label)) in items.iter().enumerate() {
            let y = item_y_start + (i as f32 * item_spacing);
            
            draw_circle(circle_x, y, circle_radius, *color);
            draw_circle_lines(circle_x, y, circle_radius, 1.5, WHITE);
            
            draw_text_ex(
                label,
                circle_x + circle_radius + 10.0,
                y + 5.0,
                TextParams {
                    font: Some(font),
                    font_size: 14,
                    color: WHITE,
                    ..Default::default()
                },
            );
        }
    }
}

pub struct ToggleButton {
    pub x: f32,
    pub y: f32,
    pub collapsed: bool,
    hovered: bool,
    pressed: bool,
}

impl ToggleButton {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            collapsed: false,
            hovered: false,
            pressed: false,
        }
    }

    pub fn update(&mut self) -> bool {
        let (mx, my) = mouse_position();
        self.hovered = mx >= self.x
            && mx <= self.x + TOGGLE_BUTTON_SIZE
            && my >= self.y
            && my <= self.y + TOGGLE_BUTTON_SIZE;

        let clicked = self.hovered && is_mouse_button_pressed(MouseButton::Left);
        self.pressed = self.hovered && is_mouse_button_down(MouseButton::Left);

        clicked
    }

    pub fn draw(&self, font: &Font) {
        let color = if self.pressed {
            Color::new(0.45, 0.45, 0.45, 1.0)
        } else if self.hovered {
            Color::new(0.35, 0.35, 0.35, 1.0)
        } else {
            Color::new(0.25, 0.25, 0.25, 1.0)
        };

        // Draw subtle shadow
        draw_rectangle(
            self.x + 2.0,
            self.y + 2.0,
            TOGGLE_BUTTON_SIZE,
            TOGGLE_BUTTON_SIZE,
            Color::new(0.0, 0.0, 0.0, 0.3),
        );

        draw_rectangle(self.x, self.y, TOGGLE_BUTTON_SIZE, TOGGLE_BUTTON_SIZE, color);
        draw_rectangle_lines(
            self.x,
            self.y,
            TOGGLE_BUTTON_SIZE,
            TOGGLE_BUTTON_SIZE,
            2.0,
            if self.hovered {
                Color::new(0.6, 0.6, 0.6, 1.0)
            } else {
                WHITE
            },
        );

        // Draw arrow
        let arrow = if self.collapsed { "►" } else { "◄" };
        let text_dims = measure_text(arrow, Some(font), 20, 1.0);
        let text_x = self.x + (TOGGLE_BUTTON_SIZE - text_dims.width) / 2.0;
        let text_y = self.y + (TOGGLE_BUTTON_SIZE + text_dims.height) / 2.0 - 2.0;

        draw_text_ex(
            arrow,
            text_x,
            text_y,
            TextParams {
                font: Some(font),
                font_size: 20,
                color: WHITE,
                ..Default::default()
            },
        );
    }
}
