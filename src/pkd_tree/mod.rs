mod batch_insert;
mod build_tree;
mod nearest_neighbors;

use crate::{
    batch_par_kd_tree_trait::{BatchParKDTree, Coord},
    bbox::BBox,
};
use num_traits::{Bounded, NumOps};
use ordered_float::OrderedFloat;
use rayon::prelude::*;

const LAMBDA: usize = 6; // number of levels in a tree skeleton
const PHI: usize = 32; // max number of points in leaf node
const SIGMA: usize = 32; // oversampling rate
const ALPHA: f32 = 0.3; // imbalance limit
const SIMPLE_THRESHOLD: usize = SIGMA * 2usize.pow(LAMBDA as u32);

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

impl<C: Coord, const K: usize> BatchParKDTree<C, K> for PKDTree<C, K> {
    fn build(mut points: Vec<[C; K]>) -> Self {
        let bbox = BBox::build(&points);
        println!("Bounded by {:?}", bbox);
        Self(Node::build(&mut points, bbox))
    }

    fn batch_insert(&mut self, mut points: Vec<[C; K]>) {
        let bbox = BBox::merge(BBox::build(&points), self.bbox());
        self.0.batch_insert(&mut points, bbox);
    }

    fn batch_nearests(&self, points: &[[C; K]], num: usize) -> Vec<Vec<(C, [C; K])>> {
        points.into_par_iter().map(|p| self.nearests(p, num)).collect()
    }
}
