use crate::batch_par_kd_tree_trait::Coord;
use rayon::prelude::*;

#[derive(Debug, Copy, Clone)]
pub struct BBox<C: Coord, const K: usize> {
    mins: [C; K],
    maxs: [C; K],
}

impl<C: Coord, const K: usize> BBox<C, K> {
    pub fn unbounded() -> Self {
        Self {
            mins: [C::min_value(); K],
            maxs: [C::max_value(); K],
        }
    }

    pub fn build(points: &[[C; K]]) -> Self {
        let (mins, maxs) = rayon::join(
            || {
                points.par_iter().map(|p| *p).reduce(
                    || [C::max_value(); K],
                    |p1, p2| std::array::from_fn(|i| C::min(p1[i], p2[i])),
                )
            },
            || {
                points.par_iter().map(|p| *p).reduce(
                    || [C::min_value(); K],
                    |p1, p2| std::array::from_fn(|i| C::max(p1[i], p2[i])),
                )
            },
        );
        Self { mins, maxs }
    }

    pub fn widest_axis(&self) -> usize {
        let (_width, axis) = (0..K).map(|i| (self.maxs[i] - self.mins[i], i)).max().unwrap();
        axis
    }

    pub fn split(&self, coord: C, axis: usize) -> (Self, Self) {
        let (mut left, mut right) = (*self, *self);
        left.maxs[axis] = coord;
        right.mins[axis] = coord;
        (left, right)
    }

    pub fn merge(b1: Self, b2: Self) -> Self {
        let mins = std::array::from_fn(|i| C::min(b1.mins[i], b2.mins[i]));
        let maxs = std::array::from_fn(|i| C::max(b1.maxs[i], b2.maxs[i]));
        Self { mins, maxs }
    }
}

impl<C: Coord, const K: usize> Default for BBox<C, K> {
    fn default() -> Self {
        Self::unbounded()
    }
}
