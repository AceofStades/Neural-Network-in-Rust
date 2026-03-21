use macroquad::prelude::*;

// Layout constants
pub const TOP_BAR_HEIGHT: f32 = 60.0;
pub const LEFT_PANEL_WIDTH_EXPANDED: f32 = 250.0;
pub const LEFT_PANEL_WIDTH_COLLAPSED: f32 = 40.0;
pub const BOTTOM_PANEL_HEIGHT: f32 = 100.0;
pub const TOGGLE_BUTTON_SIZE: f32 = 30.0;
pub const PANEL_PADDING: f32 = 15.0;

#[derive(Clone, Copy, Debug)]
pub struct LayoutDimensions {
    pub screen_width: f32,
    pub screen_height: f32,
    pub left_panel_width: f32,
    pub bottom_panel_height: f32,
    pub top_bar_height: f32,
}

impl LayoutDimensions {
    pub fn new(screen_width: f32, screen_height: f32, left_panel_collapsed: bool) -> Self {
        Self {
            screen_width,
            screen_height,
            left_panel_width: if left_panel_collapsed {
                LEFT_PANEL_WIDTH_COLLAPSED
            } else {
                LEFT_PANEL_WIDTH_EXPANDED
            },
            bottom_panel_height: BOTTOM_PANEL_HEIGHT,
            top_bar_height: TOP_BAR_HEIGHT,
        }
    }

    /// Calculate the available area for network visualization
    pub fn network_area(&self) -> NetworkArea {
        let x = self.left_panel_width;
        let y = self.top_bar_height;
        let width = self.screen_width - self.left_panel_width;
        let height = self.screen_height - self.top_bar_height - self.bottom_panel_height;

        NetworkArea {
            x,
            y,
            width,
            height,
        }
    }

    /// Get the bounds for the left panel
    pub fn left_panel_bounds(&self) -> PanelBounds {
        PanelBounds {
            x: 0.0,
            y: 0.0,
            width: self.left_panel_width,
            height: self.screen_height - self.bottom_panel_height,
        }
    }

    /// Get the bounds for the bottom panel
    pub fn bottom_panel_bounds(&self) -> PanelBounds {
        PanelBounds {
            x: 0.0,
            y: self.screen_height - self.bottom_panel_height,
            width: self.screen_width,
            height: self.bottom_panel_height,
        }
    }

    /// Get the bounds for the top bar
    pub fn top_bar_bounds(&self) -> PanelBounds {
        PanelBounds {
            x: 0.0,
            y: 0.0,
            width: self.screen_width,
            height: self.top_bar_height,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NetworkArea {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl NetworkArea {
    /// Get the center point of the network area
    pub fn center(&self) -> Vec2 {
        vec2(self.x + self.width / 2.0, self.y + self.height / 2.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PanelBounds {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl PanelBounds {
    /// Check if a point is within this panel
    pub fn contains(&self, point: Vec2) -> bool {
        point.x >= self.x
            && point.x <= self.x + self.width
            && point.y >= self.y
            && point.y <= self.y + self.height
    }

    /// Get the center point of the panel
    pub fn center(&self) -> Vec2 {
        vec2(self.x + self.width / 2.0, self.y + self.height / 2.0)
    }
}

pub struct LayoutManager {
    pub dimensions: LayoutDimensions,
    pub left_panel_collapsed: bool,
    animation_progress: f32,
    target_collapsed: bool,
}

impl LayoutManager {
    pub fn new(screen_width: f32, screen_height: f32) -> Self {
        Self {
            dimensions: LayoutDimensions::new(screen_width, screen_height, false),
            left_panel_collapsed: false,
            animation_progress: 1.0,
            target_collapsed: false,
        }
    }

    /// Toggle the left panel collapsed state
    pub fn toggle_left_panel(&mut self) {
        self.target_collapsed = !self.target_collapsed;
        self.animation_progress = 0.0;
    }

    /// Update animation and dimensions
    pub fn update(&mut self, delta_time: f32) {
        // Update screen dimensions
        let screen_w = screen_width();
        let screen_h = screen_height();

        // Animate panel collapse/expand
        if self.animation_progress < 1.0 {
            self.animation_progress += delta_time * 5.0; // 0.2 seconds animation
            self.animation_progress = self.animation_progress.min(1.0);

            if self.animation_progress >= 1.0 {
                self.left_panel_collapsed = self.target_collapsed;
            }
        }

        // Calculate current panel width with animation
        let start_width = if self.target_collapsed {
            LEFT_PANEL_WIDTH_EXPANDED
        } else {
            LEFT_PANEL_WIDTH_COLLAPSED
        };
        let end_width = if self.target_collapsed {
            LEFT_PANEL_WIDTH_COLLAPSED
        } else {
            LEFT_PANEL_WIDTH_EXPANDED
        };

        // Smooth easing function
        let t = self.animation_progress;
        let eased = if t < 0.5 {
            2.0 * t * t
        } else {
            1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
        };

        let current_width = start_width + (end_width - start_width) * eased;

        self.dimensions = LayoutDimensions {
            screen_width: screen_w,
            screen_height: screen_h,
            left_panel_width: current_width,
            bottom_panel_height: BOTTOM_PANEL_HEIGHT,
            top_bar_height: TOP_BAR_HEIGHT,
        };
    }

    /// Check if animation is in progress
    pub fn is_animating(&self) -> bool {
        self.animation_progress < 1.0
    }

    /// Get the current target state
    pub fn is_collapsed(&self) -> bool {
        self.target_collapsed
    }
}
