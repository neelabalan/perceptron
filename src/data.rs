use ndarray::prelude::*;
use ndarray::OwnedRepr;
use ndarray::ArrayBase;

pub type TargetLabels<Y> = ArrayBase<OwnedRepr<Y>, Ix1>;
pub type Features<X> = ArrayBase<OwnedRepr<X>, Ix2>;
pub type Feature<X> = ArrayBase<OwnedRepr<X>, Ix1>;
pub type Dataset<D, T, I = Ix1> = Data<ArrayBase<OwnedRepr<D>, Ix2>, ArrayBase<OwnedRepr<T>, I>>;

#[derive(Debug, Clone, PartialEq)]
pub struct Data<R, T> {
    pub records: R,
    pub targets: T,
}

impl<R, T> Data<R, T> {
    pub fn new(records: R, targets: T) -> Data<R, T> {
        Data { records, targets }
    }
}



