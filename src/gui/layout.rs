use super::theme::SCREEN_PADDING_RATIO;
use macroquad::prelude::{Vec2, vec2};

pub struct NetworkLayout {
    pub node_radius: f32,
    pub node_positions: Vec<Vec<Vec2>>,
    pub connections: Vec<(Vec2, Vec2)>,
}

pub fn calculate_layout(screen_w: f32, screen_h: f32, topology: &[usize]) -> NetworkLayout {
    if topology.is_empty() {
        return NetworkLayout {
            node_radius: 0.0,
            node_positions: vec![],
            connections: vec![],
        };
    }

    let (work_w, work_h) = calculate_work_area(screen_w, screen_h);

    let max_nodes = *topology.iter().max().unwrap_or(&1);

    let vertical_step = if max_nodes > 1 {
        work_h / (max_nodes as f32)
    } else {
        0.0
    };

    let positions = generate_node_positions(screen_w, screen_h, work_w, vertical_step, topology);
    let radius = calculate_safe_radius(work_w, work_h, vertical_step, topology, max_nodes);
    let connections = generate_connections(&positions);

    NetworkLayout {
        node_radius: radius,
        node_positions: positions,
        connections: connections,
    }
}

fn calculate_work_area(screen_w: f32, screen_h: f32) -> (f32, f32) {
    (
        screen_w * SCREEN_PADDING_RATIO,
        screen_h * SCREEN_PADDING_RATIO,
    )
}

fn generate_node_positions(
    screen_w: f32,
    screen_h: f32,
    work_w: f32,
    vertical_step: f32,
    topology: &[usize],
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

        if node_count == 1 {
            layer_nodes.push(vec2(x, center_y));
        } else {
            let group_height = (node_count as f32 - 1.0) * vertical_step;
            let start_y = center_y - (group_height / 2.0);

            for row_idx in 0..node_count {
                let y = start_y + (row_idx as f32 * vertical_step);
                layer_nodes.push(vec2(x, y));
            }
        }
        positions.push(layer_nodes);
    }

    positions
}

fn calculate_safe_radius(
    work_w: f32,
    work_h: f32,
    vertical_step: f32,
    topology: &[usize],
    max_nodes: usize,
) -> f32 {
    let num_layers = topology.len();

    let max_radius_spacing = if max_nodes > 1 {
        vertical_step / 2.5
    } else {
        100.0
    };

    let col_spacing = if num_layers > 1 {
        work_w / (num_layers as f32 - 1.0)
    } else {
        work_w
    };
    let max_radius_w = col_spacing * 0.25;

    let max_radius_relative = work_h * 0.06;

    max_radius_spacing
        .min(max_radius_w)
        .min(max_radius_relative)
        .clamp(5.0, 40.0)
}

fn generate_connections(node_positions: &[Vec<Vec2>]) -> Vec<(Vec2, Vec2)> {
    let mut lines = Vec::new();

    for i in 0..node_positions.len() - 1 {
        let current_layer = &node_positions[i];
        let next_layer = &node_positions[i + 1];

        for &start_pos in current_layer {
            for &end_pos in next_layer {
                lines.push((start_pos, end_pos));
            }
        }
    }
    lines
}
