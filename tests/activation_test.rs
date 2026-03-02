use rust_nn::nn::activation::*;

// softmax([1.0, 2.0, 3.0]) = [e^1, e^2, e^3] / (e^1+e^2+e^3)
//                          ≈ [0.09003, 0.24473, 0.66524]
#[test]
fn test_softmax_sums_to_one() {
    let input = vec![1.0, 2.0, 3.0];
    let result = softmax(input);
    let sum: f32 = result.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "softmax outputs must sum to 1, got {}",
        sum
    );
}

#[test]
fn test_softmax_all_values_in_range() {
    let input = vec![1.0, 2.0, 3.0];
    let result = softmax(input);
    for val in &result {
        assert!(
            *val >= 0.0 && *val <= 1.0,
            "softmax output out of [0, 1] range: {}",
            val
        );
    }
}

#[test]
fn test_softmax_uniform_input() {
    let input = vec![2.0, 2.0, 2.0];
    let result = softmax(input);
    let expected = 1.0 / 3.0;
    for val in &result {
        assert!(
            (val - expected).abs() < 1e-6,
            "uniform input should give equal probs, got {}",
            val
        );
    }
}

#[test]
fn test_softmax_largest_input_gets_highest_prob() {
    let input = vec![1.0, 5.0, 2.0];
    let result = softmax(input);
    let max_idx = result
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(
        max_idx, 1,
        "index 1 (input 5.0) should have the highest probability"
    );
}

#[test]
fn test_softmax_known_values() {
    let input = vec![1.0f32, 2.0, 3.0];
    let result = softmax(input);
    let expected = [0.09003_f32, 0.24473, 0.66524];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-4, "expected ~{}, got {}", e, r);
    }
}

#[test]
fn test_softmax_single_element() {
    let input = vec![42.0f32];
    let result = softmax(input);
    assert_eq!(result.len(), 1);
    assert!(
        (result[0] - 1.0).abs() < 1e-6,
        "single element softmax must be 1.0, got {}",
        result[0]
    );
}

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
