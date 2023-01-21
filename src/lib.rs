use csv::ReaderBuilder;
use ndarray::prelude::*;
use ndarray_csv::Array2Reader;
use ndarray_csv::ReadError;
use std::fs::File;

pub mod perceptron;
pub mod data;

use crate::perceptron::Perceptron;
use crate::perceptron::BinaryClassifier;
use crate::data::{Dataset};

/// Convert CSV bytes into 2D array
pub fn array_from_csv(
    csv: &str,
    has_headers: bool,
    separator: u8,
) -> Result<Array2<f64>, ReadError> {
    let file = File::open(csv).unwrap();
    let mut reader = ReaderBuilder::new()
        .has_headers(has_headers)
        .delimiter(separator)
        .from_reader(file);

    reader.deserialize_array2_dynamic()
}

fn load_breast_cancer() -> Dataset<f64, u8> {
    let array = array_from_csv("data/breast_cancer.csv", true, b',').unwrap();
    let (records, targets) = (
        array.slice(s![.., 1..31]).to_owned(),
        array.column(0).to_owned(),
    );
    Dataset::new(records, targets.map(|x| *x as u8))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier() {
		let data = load_breast_cancer();
		let mut percep = Perceptron::new();
		percep.fit(&data.records, &data.targets);
		assert!(percep.b != 0 as f64);
		assert!(percep.w[0] != 0 as f64)
		// println!("{:?}, {:?}", percep.w, percep.b);
    }

}
