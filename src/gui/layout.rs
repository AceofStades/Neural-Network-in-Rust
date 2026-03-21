use super::layout_manager::NetworkArea;
use super::theme::SCREEN_PADDING_RATIO;
use macroquad::prelude::{vec2, Vec2};

const MAX_VISIBLE_NODES: usize = 16;

pub struct NetworkLayout {
    pub node_radius: f32,
    pub node_positions: Vec<Vec<Vec2>>,
    pub node_display_info: Vec<Vec<DisplayNodeInfo>>,
    pub connections: Vec<(Vec2, Vec2)>,
    pub layer_sizes: Vec<usize>,
    pub network_area: NetworkArea,
}

pub struct DisplayNodeInfo {
    pub is_real: bool,
    pub index: usize,
}

pub fn calculate_layout(network_area: NetworkArea, topology: &[usize]) -> NetworkLayout {
    if topology.is_empty() {
        return NetworkLayout {
            node_radius: 0.0,
            node_positions: vec![],
            node_display_info: vec![],
            connections: vec![],
            layer_sizes: vec![],
            network_area,
        };
    }

    // Use the provided network area instead of calculating from screen
    let work_w = network_area.width * SCREEN_PADDING_RATIO;
    let work_h = network_area.height * SCREEN_PADDING_RATIO;

    let max_nodes = *topology.iter().max().unwrap_or(&1);
    let visible_nodes = max_nodes.min(MAX_VISIBLE_NODES);

    let vertical_step = if visible_nodes > 1 {
        work_h / (visible_nodes as f32 + 1.0)
    } else {
        0.0
    };

    let (positions, display_info) =
        generate_node_positions_with_ellipsis(&network_area, work_w, vertical_step, topology);
    let radius = calculate_safe_radius(work_w, work_h, vertical_step, topology, max_nodes);
    let connections = generate_connections(&positions);

    NetworkLayout {
        node_radius: radius,
        node_positions: positions,
        node_display_info: display_info,
        connections,
        layer_sizes: topology.to_vec(),
        network_area,
    }
}

fn generate_node_positions_with_ellipsis(
    network_area: &NetworkArea,
    work_w: f32,
    vertical_step: f32,
    topology: &[usize],
) -> (Vec<Vec<Vec2>>, Vec<Vec<DisplayNodeInfo>>) {
    let start_x = network_area.x + (network_area.width - work_w) / 2.0;
    let center_y = network_area.y + network_area.height / 2.0;
    let num_layers = topology.len();

    let col_spacing = if num_layers > 1 {
        work_w / (num_layers as f32 - 1.0)
    } else {
        0.0
    };

    let mut positions = Vec::with_capacity(num_layers);
    let mut display_info = Vec::with_capacity(num_layers);

    for (col_idx, &node_count) in topology.iter().enumerate() {
        let x = start_x + (col_idx as f32 * col_spacing);
        let mut layer_nodes = Vec::new();
        let mut layer_info = Vec::new();

        if node_count <= MAX_VISIBLE_NODES {
            if node_count == 1 {
                layer_nodes.push(vec2(x, center_y));
                layer_info.push(DisplayNodeInfo {
                    is_real: true,
                    index: 0,
                });
            } else {
                let group_height = (node_count as f32 - 1.0) * vertical_step;
                let start_y = center_y - (group_height / 2.0);

                for row_idx in 0..node_count {
                    let y = start_y + (row_idx as f32 * vertical_step);
                    layer_nodes.push(vec2(x, y));
                    layer_info.push(DisplayNodeInfo {
                        is_real: true,
                        index: row_idx,
                    });
                }
            }
        } else {
            let half = MAX_VISIBLE_NODES / 2;
            let group_height = (MAX_VISIBLE_NODES as f32) * vertical_step;
            let start_y = center_y - (group_height / 2.0);

            for i in 0..half {
                let y = start_y + (i as f32 * vertical_step);
                layer_nodes.push(vec2(x, y));
                layer_info.push(DisplayNodeInfo {
                    is_real: true,
                    index: i,
                });
            }

            let ellipsis_y = start_y + (half as f32 * vertical_step);
            layer_nodes.push(vec2(x, ellipsis_y));
            layer_info.push(DisplayNodeInfo {
                is_real: false,
                index: 0,
            });

            for i in 0..half {
                let actual_idx = node_count - half + i;
                let y = start_y + ((half + 1 + i) as f32 * vertical_step);
                layer_nodes.push(vec2(x, y));
                layer_info.push(DisplayNodeInfo {
                    is_real: true,
                    index: actual_idx,
                });
            }
        }

        positions.push(layer_nodes);
        display_info.push(layer_info);
    }

    (positions, display_info)
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
        vertical_step / 3.0
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
