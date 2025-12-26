use rust_nn::nn::cost::*;

#[test]
fn test_mse() {
    assert_eq!(9.00, mse(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]));
}

#[test]
fn test_mae() {
    assert_eq!(9.00, mae(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]));
}
