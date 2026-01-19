use ndarray::{Array1, array};
use rust_nn::nn::layer::{self, Layer};

#[test]
fn test_back_prop() {
    let mut layer = Layer::new(2, 2, layer::ActivationType::ReLU);

    layer.weights = array![[0.1, 0.2], [0.3, 0.4]];
    layer.biases = array![0.0, 0.0];

    // Forward Pass
    let input = array![1.0, 2.0];
    let _ = layer.forward(&input);

    // Backward Pass
    let output_grad = array![0.5, -0.5];
    let learning_rate = 0.1;
    let da_prev = layer.backward(output_grad, learning_rate);

    // Check Gradient
    let expected_da_prev = array![-0.05, -0.05];
    let diff_da = &da_prev - &expected_da_prev;

    assert!(
        diff_da.mapv(|x| x.abs()).sum() < 1e-6,
        "da_prev incorrect.\nGot: {}\nExpected: {}",
        da_prev,
        expected_da_prev
    );

    // Check Weight updates
    let expected_weights = array![[0.05, 0.25], [0.20, 0.50]];

    let diff_w = &layer.weights - &expected_weights;
    assert!(
        diff_w.mapv(|x| x.abs()).sum() < 1e-6,
        "Weights update incorrect.\nGot: {}\nExpected: {}",
        layer.weights,
        expected_weights
    );

    // Check Bias Updates
    let expected_biases = array![-0.05, 0.05];
    let diff_b = &layer.biases - &expected_biases;

    assert!(
        diff_b.mapv(|x| x.abs()).sum() < 1e-6,
        "Biases update incorrect.\nGot: {}\nExpected: {}",
        layer.biases,
        expected_biases
    );
}
