use rust_nn::nn::cost::*;

#[test]
fn test_mse() {
    assert_eq!(9.0, mse(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]));
}

#[test]
fn test_mae() {
    assert_eq!(3.0, mae(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]));
}

#[test]
fn test_mbe() {
    assert_eq!(-9.0, mbe(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]));
}

#[test]
fn test_huber() {
    assert_eq!(2.5, huber(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], 1.0));
}

#[test]
fn test_cce() {
    let y_true = vec![1.0, 0.0, 0.0];
    let y_pred = vec![0.7, 0.2, 0.1];
    let result = cce(y_true, y_pred);

    assert!((result - 0.35667).abs() < 1e-4);
}

#[test]
fn test_bce() {
    let y_true = vec![1.0, 0.0, 1.0];
    let y_pred = vec![0.9, 0.1, 0.8];
    let result = bce(y_true, y_pred);

    assert!((result - 0.14462).abs() < 1e-4);
}
