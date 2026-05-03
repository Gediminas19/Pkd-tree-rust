use std::iter::Sum;

use num_traits::{Bounded, NumOps};
use ordered_float::OrderedFloat;

use crate::sieve::Basic;

pub trait Coord: Basic + Default + Bounded + NumOps + Sum {}
impl Coord for i32 {}
impl Coord for u32 {}
impl Coord for OrderedFloat<f32> {}
impl<C: Coord, const K: usize> Basic for [C; K] {}

pub trait BatchParKDTree<C: Coord, const K: usize> {
    fn build(points: Vec<[C; K]>) -> Self;
    fn batch_insert(&mut self, points: Vec<[C; K]>);
    fn batch_nearests(&self, points: &[[C; K]], num: usize) -> Vec<Vec<(C, [C; K])>>;
}
