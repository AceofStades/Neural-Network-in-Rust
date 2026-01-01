use ndarray::{Array1, Array2};

enum ActivationType {
    Sigmoid,
    ReLU,
    Tanh,
    SoftPlus,
}

pub struct Layer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub activation: ActivationType,
}
