use ndarray::Array1;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Cost {
    MSE,
    MAE,
    CCE,
    BCE,
    Huber(f32),
    SCCE,
}

impl Cost {
    pub fn calc(&self, target: &Array1<f32>, prediction: &Array1<f32>) -> f32 {
        match self {
            Cost::MSE => {
                let errors = (target - prediction).mapv(|x| x.powi(2));
                errors.mean().unwrap_or(0.0)
            }
            Cost::MAE => {
                let errors = (target - prediction).mapv(|x| x.abs());
                errors.mean().unwrap_or(0.0)
            }
            Cost::CCE => {
                let epsilon = 1e-9;
                let log_pred = prediction.mapv(|x| (x + epsilon).ln());
                -(target * &log_pred).sum()
            }
            Cost::BCE => {
                let epsilon = 1e-9;
                let p = prediction.mapv(|x| x.clamp(epsilon, 1.0 - epsilon));
                let term1 = target * &p.mapv(|x| x.ln());
                let term2 = (1.0 - target) * &p.mapv(|x| (1.0 - x).ln());
                -(term1 + term2).mean().unwrap_or(0.0)
            }
            Cost::Huber(delta) => {
                let errors = (target - prediction).mapv(|x| {
                    let a = x.abs();
                    if a <= *delta {
                        0.5 * a.powi(2)
                    } else {
                        delta * (a - 0.5 * delta)
                    }
                });
                errors.mean().unwrap_or(0.0)
            }
            Cost::SCCE => {
                let epsilon = 1e-9;
                let class_idx = target[0] as usize;
                -(prediction[class_idx] + epsilon).ln()
            }
        }
    }

    pub fn prime(&self, target: &Array1<f32>, prediction: &Array1<f32>) -> Array1<f32> {
        match self {
            Cost::MSE => prediction - target,
            Cost::MAE => (prediction - target).mapv(|x| x.signum()),
            Cost::CCE => {
                let epsilon = 1e-9;
                -target / (prediction + epsilon)
            }
            Cost::BCE => {
                let epsilon = 1e-9;
                let p = prediction.mapv(|x| x.clamp(epsilon, 1.0 - epsilon));
                (p.clone() - target) / (p.clone() * (1.0 - &p))
            }
            Cost::Huber(delta) => {
                let diff = prediction - target;
                diff.mapv(|x| {
                    if x.abs() <= *delta {
                        x
                    } else {
                        delta * x.signum()
                    }
                })
            }
            Cost::SCCE => {
                let epsilon = 1e-9;
                let class_idx = target[0] as usize;
                let mut grad = Array1::zeros(prediction.len());
                grad[class_idx] = -1.0 / (prediction[class_idx] + epsilon);
                grad
            }
        }
    }
}
