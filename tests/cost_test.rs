use ndarray::arr1;
use rust_nn::nn::cost::Cost;

#[test]
fn test_mse() {
    let y_true = arr1(&[1.0, 2.0, 3.0]);
    let y_pred = arr1(&[4.0, 5.0, 6.0]);
    assert_eq!(9.0, Cost::MSE.calc(&y_true, &y_pred));
}

#[test]
fn test_mae() {
    let y_true = arr1(&[1.0, 2.0, 3.0]);
    let y_pred = arr1(&[4.0, 5.0, 6.0]);
    assert_eq!(3.0, Cost::MAE.calc(&y_true, &y_pred));
}

#[test]
fn test_huber() {
    let y_true = arr1(&[1.0, 2.0, 3.0]);
    let y_pred = arr1(&[4.0, 5.0, 6.0]);
    assert_eq!(2.5, Cost::Huber(1.0).calc(&y_true, &y_pred));
}

#[test]
fn test_cce() {
    let y_true = arr1(&[1.0, 0.0, 0.0]);
    let y_pred = arr1(&[0.7, 0.2, 0.1]);
    let result = Cost::CCE.calc(&y_true, &y_pred);
    assert!((result - 0.35667).abs() < 1e-4);
}

#[test]
fn test_bce() {
    let y_true = arr1(&[1.0, 0.0, 1.0]);
    let y_pred = arr1(&[0.9, 0.1, 0.8]);
    let result = Cost::BCE.calc(&y_true, &y_pred);
    assert!((result - 0.14462).abs() < 1e-4);
}

// SCCE target is a single-element array with the class index as a float.
// -ln(prediction[class_idx]): -ln(0.7) ≈ 0.35667
#[test]
fn test_scce_calc() {
    let y_true = arr1(&[2.0]); // true class is index 2
    let y_pred = arr1(&[0.1, 0.2, 0.7]);
    let result = Cost::SCCE.calc(&y_true, &y_pred);
    assert!((result - 0.35667).abs() < 1e-4);
}

// Gradient is -1/p[class_idx] at the true index, 0 everywhere else.
// -1/0.7 ≈ -1.42857
#[test]
fn test_scce_prime() {
    let y_true = arr1(&[2.0]); // true class is index 2
    let y_pred = arr1(&[0.1, 0.2, 0.7]);
    let grad = Cost::SCCE.prime(&y_true, &y_pred);
    assert_eq!(grad.len(), 3);
    assert!((grad[0] - 0.0).abs() < 1e-6);
    assert!((grad[1] - 0.0).abs() < 1e-6);
    assert!((grad[2] - (-1.0 / 0.7)).abs() < 1e-4);
}

// Perfect prediction: loss should be near 0.
#[test]
fn test_scce_perfect_prediction() {
    let y_true = arr1(&[1.0]); // true class is index 1
    let y_pred = arr1(&[0.0, 1.0, 0.0]);
    let result = Cost::SCCE.calc(&y_true, &y_pred);
    assert!(result < 1e-6);
}
