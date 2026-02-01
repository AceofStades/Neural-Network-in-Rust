use super::theme::SCREEN_PADDING_RATIO;
use macroquad::prelude::{Vec2, vec2};

pub struct NetworkLayout {
    pub node_radius: f32,
    pub node_positions: Vec<Vec<Vec2>>,
}

pub fn calculate_layout(screen_w: f32, screen_h: f32, topology: &[usize]) -> NetworkLayout {
    if topology.is_empty() {
        return NetworkLayout {
            node_radius: 0.0,
            node_positions: vec![],
        };
    }

    let (work_w, work_h) = calculate_work_area(screen_w, screen_h);

    let radius = calculate_optimal_radius(work_w, work_h, topology);

    let positions = generate_node_positions(screen_w, screen_h, work_w, topology, radius);

    NetworkLayout {
        node_radius: radius,
        node_positions: positions,
    }
}

fn calculate_work_area(screen_w: f32, screen_h: f32) -> (f32, f32) {
    (
        screen_w * SCREEN_PADDING_RATIO,
        screen_h * SCREEN_PADDING_RATIO,
    )
}

fn calculate_optimal_radius(work_w: f32, work_h: f32, topology: &[usize]) -> f32 {
    let max_nodes_in_col = *topology.iter().max().unwrap_or(&1) as f32;
    let num_layers = topology.len() as f32;

    // Vertical Constraint: Nodes must fit in the height
    // We multiply by 2.5 to account for diameter (2.0) + gap (0.5)
    let max_radius_h = work_h / (max_nodes_in_col * 2.5);

    // Horizontal Constraint: Layers must fit in the width
    let col_spacing = if num_layers > 1.0 {
        work_w / (num_layers - 1.0)
    } else {
        0.0
    };
    let max_radius_w = if num_layers > 1.0 {
        col_spacing * 0.3
    } else {
        max_radius_h
    };

    // Absolute Constraint: Never bigger than 30px, never smaller than 2px
    max_radius_h.min(max_radius_w).clamp(2.0, 30.0)
}

fn generate_node_positions(
    screen_w: f32,
    screen_h: f32,
    work_w: f32,
    topology: &[usize],
    radius: f32,
) -> Vec<Vec<Vec2>> {
    let start_x = (screen_w - work_w) / 2.0;
    let center_y = screen_h / 2.0;
    let num_layers = topology.len();

    let col_spacing = if num_layers > 1 {
        work_w / (num_layers as f32 - 1.0)
    } else {
        0.0
    };

    let mut positions = Vec::with_capacity(num_layers);

    for (col_idx, &node_count) in topology.iter().enumerate() {
        let x = start_x + (col_idx as f32 * col_spacing);
        let mut layer_nodes = Vec::with_capacity(node_count);

        let col_height = (node_count as f32) * (radius * 2.5);
        let start_y = center_y - (col_height / 2.0) + (radius * 1.25);

        for row_idx in 0..node_count {
            let y = start_y + (row_idx as f32 * (radius * 2.5));
            layer_nodes.push(vec2(x, y));
        }
        positions.push(layer_nodes);
    }

    positions
}
