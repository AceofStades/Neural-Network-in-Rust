use rust_nn::nn::activation::*;

#[test]
fn test_relu_pos() {
    let input = 10.0;
    let expected = 10.0;
    let result = relu(input);

    assert_eq!(result, expected);
}

#[test]
fn test_relu_neg() {
    let input = -5.0;
    let result = relu(input);

    assert_eq!(result, 0.0);
}

#[test]
fn test_sigmoid() {
    let input = 0.0;
    let result = sigmoid(input);

    assert_eq!(result, 0.5);
}

#[test]
fn test_tanh() {
    let input = 0.0;
    let result = tanh(input);

    assert_eq!(result, 0.0);
}
