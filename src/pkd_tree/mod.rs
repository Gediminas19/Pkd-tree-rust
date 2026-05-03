mod batch_insert;
mod build_tree;

use crate::sieve::Basic;
use num_traits::{Bounded, NumOps};
use ordered_float::OrderedFloat;
use rayon::prelude::*;

const LAMBDA: usize = 6; // number of levels in a tree skeleton
const PHI: usize = 32; // max number of points in leaf node
const SIGMA: usize = 32; // oversampling rate
const ALPHA: f32 = 0.3; // imbalance limit
const SIMPLE_THRESHOLD: usize = SIGMA * 2usize.pow(LAMBDA as u32);

pub trait Coord: Basic + Default + Bounded + NumOps {}
impl Coord for i32 {}
impl Coord for OrderedFloat<f32> {}

impl<C: Coord, const K: usize> Basic for [C; K] {}

#[derive(Debug, Copy, Clone)]
struct BBox<C: Coord, const K: usize> {
    mins: [C; K],
    maxs: [C; K],
}

impl<C: Coord, const K: usize> BBox<C, K> {
    fn unbounded() -> Self {
        Self {
            mins: [C::min_value(); K],
            maxs: [C::max_value(); K],
        }
    }

    fn build(points: &[[C; K]]) -> Self {
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

    fn widest_axis(&self) -> usize {
        let (_width, axis) = (0..K).map(|i| (self.maxs[i] - self.mins[i], i)).max().unwrap();
        axis
    }

    fn split(&self, coord: C, axis: usize) -> (Self, Self) {
        let (mut left, mut right) = (*self, *self);
        left.maxs[axis] = coord;
        right.mins[axis] = coord;
        (left, right)
    }

    fn merge(b1: Self, b2: Self) -> Self {
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

#[derive(Debug, Clone)]
enum Node<C: Coord, const K: usize> {
    Leaf {
        bbox: BBox<C, K>,
        points: Vec<[C; K]>,
    },
    Interior {
        size: usize,
        splitter: (C, usize), // usize says which dimension we are splitting on
        bbox: BBox<C, K>,
        left: Box<Node<C, K>>,
        right: Box<Node<C, K>>,
    },
}

impl<C: Coord, const K: usize> Node<C, K> {
    fn size(&self) -> usize {
        match self {
            Node::Leaf { points, .. } => points.len(),
            Node::Interior { size, .. } => *size,
        }
    }

    fn bbox(&self) -> BBox<C, K> {
        match self {
            Node::Leaf { bbox, .. } => *bbox,
            Node::Interior { bbox, .. } => *bbox,
        }
    }
}

impl<C: Coord, const K: usize> Default for Node<C, K> {
    fn default() -> Self {
        Self::Leaf {
            bbox: BBox::unbounded(),
            points: vec![],
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct BucketInfo<C: Coord, const K: usize> {
    size: usize,
    depth: usize,
    bbox: BBox<C, K>,
}

#[derive(Debug, Default, Clone)]
enum SkeletonEntry<C: Coord> {
    #[default]
    Blank,
    Bucket(usize),
    Splitter {
        total: usize,
        split: usize,
        splitter: (C, usize),
    },
}

// the first LAMBDA layers of a PKD tree
#[derive(Debug, Clone)]
struct Skeleton<C: Coord> {
    n_buckets: usize,
    flattened: Vec<SkeletonEntry<C>>, // skeleton flattened in heap-tree style for fast bucket lookup
                                      // skel_tree: SkeletonNode<Node<C, K>>,
}

#[derive(Debug, Clone)]
pub struct PKDTree<C: Coord, const K: usize>(Node<C, K>);

impl<C: Coord, const K: usize> PKDTree<C, K> {
    fn bbox(&self) -> BBox<C, K> {
        self.0.bbox()
    }
}
