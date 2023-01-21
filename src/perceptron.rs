use ndarray::prelude::*;
use ndarray::Zip;

use crate::data::{Feature, Features, TargetLabels};

pub trait BinaryClassifier<M, N> {
    fn fit(&mut self, x: &Features<M>, y: &TargetLabels<N>);
    fn predict(&self, x: &Feature<M>) -> u8;
    fn score(&self, y: &TargetLabels<N>, y_pred: &TargetLabels<N>) -> f64;
}

#[derive(Debug, Clone, PartialEq)]
pub struct Perceptron {
    pub w: Array1<f64>,
    pub b: f64,
    pub epoch: u8,
}
impl Perceptron {
    fn model(&self, feature: &Array1<f64>) -> u8 {
        (self.w.dot(feature) as f64 >= self.b) as u8
    }
    pub fn new() -> Self {
        Self {
            w: Array1::ones(1),
            b: 0.0,
            epoch: 1,
        }
    }
}

impl BinaryClassifier<f64, u8> for Perceptron {
    fn fit(&mut self, x: &Features<f64>, y: &TargetLabels<u8>) {
        self.w = Array1::ones(x.shape()[1]);
        self.b = 0.0;
        for ep in 0..self.epoch {
            Zip::from(x.rows()).and(y).for_each(|feature, &label| {
                let y_pred = self.model(&feature.to_owned());
                if label == 1 && y_pred == 0 {
                    self.w += &feature;
                    self.b += 1.0;
                } else {
                    self.w -= &feature;
                    self.b -= 1.0;
                }
            });
			let y_preds: TargetLabels<u8> = x.rows().into_iter().map(|feature| self.predict(&feature.to_owned())).collect();
			println!("accuracy_score={}", self.score(y, &y_preds))
        }
    }
    fn predict(&self, x: &Feature<f64>) -> u8 {
        self.model(x)
    }
    fn score(&self, y: &TargetLabels<u8>, y_pred: &TargetLabels<u8>) -> f64 {
        let mut trues = 0;
        for (a, b) in y.iter().zip(y_pred.iter()) {
            if a == b {
                trues += 1;
            }
        }
        trues as f64 / y.len() as f64
    }
}
