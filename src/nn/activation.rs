use std::f64::consts;

const E: f32 = consts::E as f32;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

pub fn tanh(x: f32) -> f32 {
    2.0 / (1.0 + E.powf(-2.0 * x)) - 1.0
}

pub fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 as f32 }
}

// pub fn softmax(x: f32) -> f32 {}

pub fn softplus(x: f32) -> f32 {
    (1.0 + E.powf(x)).log(E)
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
