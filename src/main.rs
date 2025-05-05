use ndarray::{Array2, Array1, s};
use std::fs::File;
use csv::ReaderBuilder;
use std::{error::Error, io::Write, sync::{atomic::{AtomicBool, Ordering}, Arc}, time::Duration};
use std::thread;
use linfa::dataset::Dataset;
use std::collections::HashMap;

// Multi-Layer Neural Network (Multi-class)
struct MultiLayerNN {
    w1: Array2<f64>,
    b1: Array1<f64>,
    w2: Array2<f64>,
    b2: Array1<f64>,
    learning_rate: f64,
}

impl MultiLayerNN {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let w1 = Array2::from_shape_fn((input_size, hidden_size), |_| rand::random::<f64>() * 0.1);
        let b1 = Array1::zeros(hidden_size);
        let w2 = Array2::from_shape_fn((hidden_size, output_size), |_| rand::random::<f64>() * 0.1);
        let b2 = Array1::zeros(output_size);

        Self {
            w1,
            b1,
            w2,
            b2,
            learning_rate,
        }
    }

    fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    fn relu_deriv(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }

    fn forward(&self, input: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
        let z1 = input.dot(&self.w1) + &self.b1;
        let a1 = z1.mapv(Self::relu);

        let z2 = a1.dot(&self.w2) + &self.b2;
        let a2 = z2.mapv(Self::relu);

        (a1, a2)
    }

    fn train(&mut self, inputs: &Array2<f64>, targets: &Array2<f64>, epochs: usize) {
        for epoch in 0..epochs {
            for (input, target) in inputs.outer_iter().zip(targets.outer_iter()) {
                let input = input.to_owned();
                let target = target.to_owned();

                // Forward pass
                let (a1, a2) = self.forward(&input);

                // Backward pass
                let output_error = &a2 - &target;
                let output_deriv = a2.mapv(Self::relu_deriv);
                let output_delta = &output_deriv * &output_error;

                let hidden_error = self.w2.dot(&output_delta);
                let hidden_deriv = a1.mapv(Self::relu_deriv);
                let hidden_delta = &hidden_deriv * &hidden_error;

                // Gradient descent updates
                self.w2 = &self.w2 - &(a1.view().insert_axis(ndarray::Axis(1))
                    .dot(&output_delta.view().insert_axis(ndarray::Axis(0))) * self.learning_rate);
                self.b2 = &self.b2 - &(output_delta.clone() * self.learning_rate);

                self.w1 = &self.w1 - &(input.view().insert_axis(ndarray::Axis(1))
                    .dot(&hidden_delta.view().insert_axis(ndarray::Axis(0))) * self.learning_rate);
                self.b1 = &self.b1 - &(hidden_delta * self.learning_rate);
            }

            // Optional: Print epoch information for debugging
            println!("Epoch {}/{} completed", epoch + 1, epochs);
        }
    }

    fn predict(&self, input: &Array1<f64>) -> usize {
        let (_, a2) = self.forward(input);
        a2.iter().cloned().enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "data/Rice_MSC_Dataset_sample.csv";
    let file = File::open(&file_path)?;

    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let mut records = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let solidity: f64 = record[0].parse()?;
        let aspect_ratio: f64 = record[1].parse()?;
        let roundness: f64 = record[2].parse()?;
        let compactness: f64 = record[3].parse()?;
        let class = record[4].to_string();
        records.push((solidity, aspect_ratio, roundness, compactness, class));
    }

    let mut features = Array2::<f64>::zeros((records.len(), 4));
    let mut labels_idx = Vec::new();

    let mut class_map = HashMap::new();
    let mut class_counter = 0;

    for (i, (solidity, aspect_ratio, roundness, compactness, class)) in records.iter().enumerate() {
        features[[i, 0]] = *solidity;
        features[[i, 1]] = *aspect_ratio;
        features[[i, 2]] = *roundness;
        features[[i, 3]] = *compactness;

        let class_idx = *class_map.entry(class.clone()).or_insert_with(|| {
            let idx = class_counter;
            class_counter += 1;
            idx
        });
        labels_idx.push(class_idx);
    }

    let num_classes = class_counter;
    let mut labels = Array2::<f64>::zeros((labels_idx.len(), num_classes));
    for (i, &class_idx) in labels_idx.iter().enumerate() {
        labels[[i, class_idx]] = 1.0;
    }

    println!("Fitur ({} sampel, {} fitur):", features.nrows(), features.ncols());
    println!("{:?}", features.slice(s![.., ..]));
    println!("Class map: {:?}", class_map);

    let (train, test) = Dataset::new(features.clone(), labels.clone()).split_with_ratio(0.8);

    let running_nn = Arc::new(AtomicBool::new(true));
    let running_nn_clone = running_nn.clone();

    let handle_nn = thread::spawn(move || {
        let mut dots = 0;
        while running_nn_clone.load(Ordering::Relaxed) {
            print!("\rTraining Neural Network{}   ", ".".repeat(dots));
            dots = (dots + 1) % 4;
            std::io::stdout().flush().unwrap();
            thread::sleep(Duration::from_millis(500));
        }
        println!("\rTraining Neural Network selesai!        ");
    });

    let mut nn = MultiLayerNN::new(4, 64, num_classes, 0.01); // Increase hidden layer size and learning rate
    nn.train(train.records(), train.targets(), 5000); // Increase epochs

    running_nn.store(false, Ordering::Relaxed);
    handle_nn.join().unwrap();

    let mut predictions = Vec::new();
    let mut actuals = Vec::new();

    for (sample, target_row) in test.records().outer_iter().zip(test.targets().outer_iter()) {
        let pred_idx = nn.predict(&sample.to_owned());
        let actual_idx = target_row.iter().cloned().enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
        predictions.push(pred_idx);
        actuals.push(actual_idx);
    }

    println!("Neural Network Predictions:");
    for (i, (pred, actual)) in predictions.iter().zip(actuals.iter()).enumerate() {
        println!("Sample {}: Predicted {}, Actual {}", i, pred, actual);
    }

    let accuracy = predictions.iter().zip(actuals.iter()).filter(|(p, a)| p == a).count() as f64 / predictions.len() as f64;
    println!("Accuracy: {:.2}%", accuracy * 100.0);

    Ok(())
}
