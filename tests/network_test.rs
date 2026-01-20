use ndarray::{arr1, arr2};
use rust_nn::nn::layer::{ActivationType, Layer};
use rust_nn::nn::network::Network;

#[test]
fn test_network_wiring_forward_pass() {
    let mut net = Network::new();

    let mut l1 = Layer::new(2, 2, ActivationType::ReLU);
    l1.weights = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    l1.biases = arr1(&[0.0]);
    net.add(l1);

    let mut l2 = Layer::new(2, 1, ActivationType::ReLU);
    l2.weights = arr2(&[[1.0], [1.0]]);
    l2.biases = arr1(&[0.0]);
    net.add(l2);

    let input = arr1(&[10.0, 20.0]);
    let output = net.predict(input);

    assert!(
        (output[0] - 30.0).abs() < 1e-4,
        "Network failed to wire layers correctly"
    );
}

#[test]
fn test_network_learning_xor() {
    let inputs = vec![
        arr1(&[0.0, 0.0]),
        arr1(&[0.0, 1.0]),
        arr1(&[1.0, 0.0]),
        arr1(&[1.0, 1.0]),
    ];

    let targets = vec![arr1(&[0.0]), arr1(&[1.0]), arr1(&[1.0]), arr1(&[0.0])];

    let mut net = Network::new();
    net.add(Layer::new(2, 4, ActivationType::Tanh));
    net.add(Layer::new(4, 1, ActivationType::Sigmoid));

    let learning_rate = 0.2;

    for epoch in 0..10000 {
        net.train_one_epoch(&inputs, &targets, learning_rate);

        if epoch % 1000 == 0 {
            let mut total_error = 0.0;
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let pred = net.predict(input.clone());
                total_error += (pred[0] - target[0]).powi(2);
            }
            println!("Epoch {}: Total Error = {:.4}", epoch, total_error);
        }
    }

    let p1 = net.predict(inputs[0].clone())[0];
    let p2 = net.predict(inputs[1].clone())[0];
    let p3 = net.predict(inputs[2].clone())[0];
    let p4 = net.predict(inputs[3].clone())[0];

    println!(
        "Final Results: [0,0]->{:.2}, [0,1]->{:.2}, [1,0]->{:.2}, [1,1]->{:.2}",
        p1, p2, p3, p4
    );

    assert!(p1 < 0.2);
    assert!(p2 > 0.8);
    assert!(p3 > 0.8);
    assert!(p4 < 0.2);
}
