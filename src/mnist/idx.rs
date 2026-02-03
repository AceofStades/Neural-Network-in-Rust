use ::idx_rs::Idx;
use ndarray::Array1;
use std::fs::File;
use std::io::BufReader;

pub struct MnistDataset {
    pub train_images: Vec<Array1<f32>>,
    pub train_labels: Vec<Array1<f32>>,
    pub test_images: Vec<Array1<f32>>,
    pub test_labels: Vec<Array1<f32>>,
}

impl MnistDataset {
    pub fn load(base_path: &str) -> Self {
        println!("Loading MNIST dataset from {}...", base_path);

        let train_img_path = format!("{}/train-images.idx3-ubyte", base_path);
        let train_lbl_path = format!("{}/train-labels.idx1-ubyte", base_path);
        let test_img_path = format!("{}/t10k-images.idx3-ubyte", base_path);
        let test_lbl_path = format!("{}/t10k-labels.idx1-ubyte", base_path);

        Self {
            train_images: parse_images(&train_img_path),
            train_labels: parse_labels(&train_lbl_path),
            test_images: parse_images(&test_img_path),
            test_labels: parse_labels(&test_lbl_path),
        }
    }
}

fn parse_images(path: &str) -> Vec<Array1<f32>> {
    let f = File::open(path).expect("failed to open image file");
    let reader = BufReader::new(f);

    // Use idx-rs to parse the file
    let idx: Idx<u8> = Idx::new(reader).expect("failed to parse idx file");

    // IDX format shape: [number_of_images, rows, cols]
    // We want to flatten rows * cols -> 784
    let rows = idx.shape()[1];
    let cols = idx.shape()[2];
    let size = rows * cols; // Should be 784

    // Iterate over chunks (each chunk is one image)
    idx.data()
        .chunks(size)
        .map(|chunk: &[u8]| {
            let data: Vec<f32> = chunk.iter().map(|&p| p as f32 / 255.0).collect();
            Array1::from(data)
        })
        .collect()
}

fn parse_labels(path: &str) -> Vec<Array1<f32>> {
    let f = File::open(path).expect("failed to open label file");
    let reader = BufReader::new(f);

    let idx: Idx<u8> = Idx::new(reader).expect("failed to parse idx file");

    idx.data()
        .iter()
        .map(|&label| {
            // One-hot encoding
            // Label 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            let mut one_hot = Array1::zeros(10);
            one_hot[label as usize] = 1.0;
            one_hot
        })
        .collect()
}
