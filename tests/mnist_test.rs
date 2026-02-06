use rust_nn::mnist::parser::MnistDataset;

#[test]
fn test_mnist_loading() {
    let mnist = MnistDataset::load("mnist-dataset");

    assert_eq!(
        mnist.train_images.len(),
        60_000,
        "Should have 60k training images"
    );
    assert_eq!(
        mnist.train_labels.len(),
        60_000,
        "Should have 60k training labels"
    );
    assert_eq!(
        mnist.test_images.len(),
        10_000,
        "Should have 10k test images"
    );

    let first_image = &mnist.train_images[0];
    assert_eq!(
        first_image.len(),
        784,
        "Image should be flattened to 784 pixels (28x28)"
    );

    let max_val = first_image.iter().fold(0.0f32, |a, &b| a.max(b));
    let min_val = first_image.iter().fold(1.0f32, |a, &b| a.min(b));

    assert!(max_val <= 1.0, "Pixel values should be normalized <= 1.0");
    assert!(min_val >= 0.0, "Pixel values should be normalized >= 0.0");

    let first_label = &mnist.train_labels[0];
    assert_eq!(first_label.len(), 10, "Label should be a vector of size 10");

    let sum: f32 = first_label.sum();
    assert!(
        (sum - 1.0).abs() < f32::EPSILON,
        "Label should be one-hot encoded (sum = 1.0)"
    );
}
