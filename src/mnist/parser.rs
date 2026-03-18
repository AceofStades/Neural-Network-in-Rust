use ndarray::Array1;
use std::fs::File;
use std::io::{BufReader, Read};

pub struct MnistDataset {
    pub train_images: Vec<Array1<f32>>,
    pub train_labels: Vec<Array1<f32>>,
    pub test_images: Vec<Array1<f32>>,
    pub test_labels: Vec<Array1<f32>>,
}

impl MnistDataset {
    pub fn load(base_path: &str) -> Self {
        println!("Loading MNIST dataset from {}...", base_path);

        let dataset = Self {
            train_images: parse_images(&format!("{}/train-images.idx3-ubyte", base_path)),
            train_labels: parse_labels(&format!("{}/train-labels.idx1-ubyte", base_path)),
            test_images: parse_images(&format!("{}/t10k-images.idx3-ubyte", base_path)),
            test_labels: parse_labels(&format!("{}/t10k-labels.idx1-ubyte", base_path)),
        };

        println!(
            "MNIST dataset loaded: {} training samples, {} test samples",
            dataset.train_images.len(),
            dataset.test_images.len()
        );

        dataset
    }
}

fn parse_images(path: &str) -> Vec<Array1<f32>> {
    let mut reader = BufReader::new(File::open(path).expect("failed to open image file"));

    let magic = read_u32(&mut reader);
    let count = read_u32(&mut reader);
    let rows = read_u32(&mut reader);
    let cols = read_u32(&mut reader);

    assert_eq!(
        magic, 2051,
        "Invalid magic number for Images (expected 2051)"
    );

    let size = (rows * cols) as usize;
    let mut images = Vec::with_capacity(count as usize);

    let mut buffer = vec![0u8; size];

    for _ in 0..count {
        reader
            .read_exact(&mut buffer)
            .expect("Failed to read image data");

        let data: Vec<f32> = buffer.iter().map(|&p| p as f32 / 255.0).collect();
        images.push(Array1::from(data));
    }

    images
}

fn parse_labels(path: &str) -> Vec<Array1<f32>> {
    let mut reader = BufReader::new(File::open(path).expect("failed to open label file"));

    let magic = read_u32(&mut reader);
    let count = read_u32(&mut reader);

    assert_eq!(
        magic, 2049,
        "Invalid magic number for Labels (expected 2049)"
    );

    let mut labels = Vec::with_capacity(count as usize);
    let mut buffer = vec![0u8; count as usize];

    reader
        .read_exact(&mut buffer)
        .expect("Failed to read label data");

    for &label in buffer.iter() {
        let mut one_hot = Array1::zeros(10);
        one_hot[label as usize] = 1.0;
        labels.push(one_hot);
    }

    labels
}

fn read_u32(reader: &mut impl Read) -> u32 {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).expect("Failed to read u32");
    u32::from_be_bytes(buf)
}
