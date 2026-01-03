use ndarray::{Array1, arr1, arr2};
use rust_nn::nn::layer::*;

fn assert_arrays_approx_equal(a: &Array1<f32>, b: &Array1<f32>) {
    let epsilon = 1e-4;
    let diff = (a - b).mapv(f32::abs);
    if diff.iter().any(|&x| x > epsilon) {
        panic!("Arrays not equal!\nLeft: {:?}\nRight: {:?}", a, b);
    }
}
#[test]
fn forw_prop_relu() {
    let layer: Layer = Layer {
        weights: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
        biases: arr1(&[1.0, -5.0]),
        activation: ActivationType::ReLU,
    };

    let input = arr1(&[1.0, 1.0]);
    let result = layer.forward(&input);
    let expected = arr1(&[2.0, 0.0]);

    assert_arrays_approx_equal(&result, &expected);
}

#[test]
fn forw_prop_sigmoid() {
    let layer: Layer = Layer {
        weights: arr2(&[[100.0]]),
        biases: arr1(&[0.0]),
        activation: ActivationType::Sigmoid,
    };

    let input = arr1(&[0.0]);
    let result = layer.forward(&input);

    assert_eq!(result[0], 0.5);
}
