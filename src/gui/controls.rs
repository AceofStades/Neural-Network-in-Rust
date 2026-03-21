use macroquad::prelude::*;

#[derive(Debug, Clone)]
pub enum ControlMessage {
    Pause,
    Resume,
    Reset,
    SetLearningRate(f32),
    SetSpeed(f32),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrainingState {
    Running,
    Paused,
}

pub struct Button {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub label: String,
    pub hovered: bool,
    pub pressed: bool,
}

impl Button {
    pub fn new(x: f32, y: f32, width: f32, height: f32, label: &str) -> Self {
        Self {
            x,
            y,
            width,
            height,
            label: label.to_string(),
            hovered: false,
            pressed: false,
        }
    }

    pub fn update(&mut self) -> bool {
        let (mx, my) = mouse_position();
        self.hovered = mx >= self.x
            && mx <= self.x + self.width
            && my >= self.y
            && my <= self.y + self.height;

        let clicked = self.hovered && is_mouse_button_pressed(MouseButton::Left);
        self.pressed = self.hovered && is_mouse_button_down(MouseButton::Left);

        clicked
    }

    pub fn draw(&self, font: &Font) {
        let color = if self.pressed {
            Color::new(0.3, 0.6, 0.3, 1.0)
        } else if self.hovered {
            Color::new(0.4, 0.7, 0.4, 1.0)
        } else {
            Color::new(0.3, 0.5, 0.3, 1.0)
        };

        draw_rectangle(self.x, self.y, self.width, self.height, color);
        draw_rectangle_lines(self.x, self.y, self.width, self.height, 2.0, WHITE);

        let text_dims = measure_text(&self.label, Some(font), 18, 1.0);
        let text_x = self.x + (self.width - text_dims.width) / 2.0;
        let text_y = self.y + (self.height + text_dims.height) / 2.0 - 2.0;

        draw_text_ex(
            &self.label,
            text_x,
            text_y,
            TextParams {
                font: Some(font),
                font_size: 18,
                color: WHITE,
                ..Default::default()
            },
        );
    }
}

pub struct Slider {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub label: String,
    pub min: f32,
    pub max: f32,
    pub value: f32,
    pub dragging: bool,
}

impl Slider {
    pub fn new(x: f32, y: f32, width: f32, label: &str, min: f32, max: f32, initial: f32) -> Self {
        Self {
            x,
            y,
            width,
            height: 20.0,
            label: label.to_string(),
            min,
            max,
            value: initial.clamp(min, max),
            dragging: false,
        }
    }

    pub fn update(&mut self) -> bool {
        let (mx, my) = mouse_position();
        let handle_x = self.x + (self.value - self.min) / (self.max - self.min) * self.width;
        let handle_radius = 10.0;

        let over_handle = (mx - handle_x).abs() < handle_radius
            && (my - self.y).abs() < handle_radius;

        if is_mouse_button_pressed(MouseButton::Left) && over_handle {
            self.dragging = true;
        }

        if is_mouse_button_released(MouseButton::Left) {
            self.dragging = false;
        }

        let mut changed = false;
        if self.dragging {
            let t = ((mx - self.x) / self.width).clamp(0.0, 1.0);
            let new_value = self.min + t * (self.max - self.min);
            if (new_value - self.value).abs() > 0.001 {
                self.value = new_value;
                changed = true;
            }
        }

        changed
    }

    pub fn draw(&self, font: &Font) {
        // Label
        let label_y = self.y - 5.0;
        draw_text_ex(
            &self.label,
            self.x,
            label_y,
            TextParams {
                font: Some(font),
                font_size: 16,
                color: WHITE,
                ..Default::default()
            },
        );

        // Track
        draw_rectangle(
            self.x,
            self.y,
            self.width,
            self.height,
            Color::new(0.3, 0.3, 0.3, 1.0),
        );
        draw_rectangle_lines(self.x, self.y, self.width, self.height, 2.0, WHITE);

        // Handle
        let handle_x = self.x + (self.value - self.min) / (self.max - self.min) * self.width;
        let handle_color = if self.dragging {
            Color::new(0.4, 0.8, 0.4, 1.0)
        } else {
            Color::new(0.5, 0.7, 0.5, 1.0)
        };

        draw_circle(handle_x, self.y + self.height / 2.0, 10.0, handle_color);
        draw_circle_lines(handle_x, self.y + self.height / 2.0, 10.0, 2.0, WHITE);

        // Value display
        let value_text = format!("{:.3}", self.value);
        let value_x = self.x + self.width + 10.0;
        draw_text_ex(
            &value_text,
            value_x,
            self.y + self.height / 2.0 + 5.0,
            TextParams {
                font: Some(font),
                font_size: 16,
                color: WHITE,
                ..Default::default()
            },
        );
    }
}

pub struct ControlPanel {
    pub play_pause_button: Button,
    pub reset_button: Button,
    pub learning_rate_slider: Slider,
    pub speed_slider: Slider,
    pub training_state: TrainingState,
}

impl ControlPanel {
    pub fn new(_screen_width: f32, screen_height: f32, initial_lr: f32) -> Self {
        let panel_y = screen_height - 250.0;
        let button_width = 100.0;
        let button_height = 40.0;
        let spacing = 20.0;

        let play_pause_x = spacing;
        let reset_x = play_pause_x + button_width + spacing;

        let slider_y = panel_y + button_height + 30.0;
        let slider_width = 200.0;

        Self {
            play_pause_button: Button::new(
                play_pause_x,
                panel_y,
                button_width,
                button_height,
                "Pause",
            ),
            reset_button: Button::new(reset_x, panel_y, button_width, button_height, "Reset"),
            learning_rate_slider: Slider::new(
                spacing,
                slider_y,
                slider_width,
                "Learning Rate",
                0.001,
                1.0,
                initial_lr,
            ),
            speed_slider: Slider::new(
                spacing,
                slider_y + 50.0,
                slider_width,
                "Speed (delay ms)",
                0.0,
                100.0,
                0.0,
            ),
            training_state: TrainingState::Running,
        }
    }

    pub fn update(&mut self) -> Vec<ControlMessage> {
        let mut messages = Vec::new();

        if self.play_pause_button.update() {
            match self.training_state {
                TrainingState::Running => {
                    messages.push(ControlMessage::Pause);
                    self.training_state = TrainingState::Paused;
                    self.play_pause_button.label = "Resume".to_string();
                }
                TrainingState::Paused => {
                    messages.push(ControlMessage::Resume);
                    self.training_state = TrainingState::Running;
                    self.play_pause_button.label = "Pause".to_string();
                }
            }
        }

        if self.reset_button.update() {
            messages.push(ControlMessage::Reset);
            self.training_state = TrainingState::Running;
            self.play_pause_button.label = "Pause".to_string();
        }

        if self.learning_rate_slider.update() {
            messages.push(ControlMessage::SetLearningRate(
                self.learning_rate_slider.value,
            ));
        }

        if self.speed_slider.update() {
            messages.push(ControlMessage::SetSpeed(self.speed_slider.value));
        }

        messages
    }

    pub fn draw(&self, font: &Font) {
        self.play_pause_button.draw(font);
        self.reset_button.draw(font);
        self.learning_rate_slider.draw(font);
        self.speed_slider.draw(font);
    }
}
