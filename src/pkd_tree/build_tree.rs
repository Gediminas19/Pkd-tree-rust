use rand::prelude::*;
use rayon::prelude::*;

use super::{LAMBDA, Node, PHI, PKDTree, SIGMA};
use crate::{
    pkd_tree::{BBox, Coord, SIMPLE_THRESHOLD, Skeleton, SkeletonEntry},
    sieve::Sieve,
    utils::kth,
};

impl<C: Coord, const K: usize> Node<C, K> {
    // shared helper, constructs balanced PKD tree (raw root node) in parallel (naively) from points
    pub fn build_simple(points: &mut [[C; K]], levels: Option<usize>, bbox: BBox<C, K>) -> Self {
        // base case
        let n = points.len();
        if let Some(level) = levels {
            if level == 0 {
                return Node::Leaf { bbox, points: vec![] };
            }
        } else if n <= PHI {
            return Node::Leaf {
                bbox,
                points: points.to_vec(),
            };
        }

        // select split axis either by widest, or cycling through dimensions
        let axis = bbox.widest_axis();

        // partition based on median
        let mid = n / 2;
        let median = kth(points, mid, &|p1, p2| p1[axis].cmp(&p2[axis]))[axis];
        let (left, right) = points.split_at_mut(mid);

        // recurse in parallel
        let (lbbox, rbbox) = bbox.split(median, axis);
        let next_level = levels.map(|l| l - 1);
        let (left, right) = rayon::join(
            || Self::build_simple(left, next_level, lbbox),
            || Self::build_simple(right, next_level, rbbox),
        );
        Node::Interior {
            size: n,
            bbox,
            splitter: (median, axis),
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    fn build_helper(points: &mut [[C; K]], size_skel: &[SkeletonEntry<C>], skel_id: usize, bbox: BBox<C, K>) -> Self {
        match &size_skel[skel_id] {
            SkeletonEntry::Bucket(_) => Self::build(points, bbox),
            SkeletonEntry::Splitter {
                total,
                split,
                splitter: (median, axis),
            } => {
                let (l_points, r_points) = points.split_at_mut(*split);
                let (lbbox, rbbox) = bbox.split(*median, *axis);
                let (left, right) = rayon::join(
                    || Self::build_helper(l_points, size_skel, 2 * skel_id, lbbox),
                    || Self::build_helper(r_points, size_skel, 2 * skel_id + 1, rbbox),
                );
                debug_assert!(*total == left.size() + right.size());
                Node::Interior {
                    size: *total,
                    splitter: (*median, *axis),
                    bbox: BBox::merge(left.bbox(), right.bbox()),
                    left: Box::new(left),
                    right: Box::new(right),
                }
            }
            SkeletonEntry::Blank => unreachable!("illegal"),
        }
    }

    pub fn build(points: &mut [[C; K]], bbox: BBox<C, K>) -> Self {
        const SAMPLE_SIZE: usize = 2usize.pow(LAMBDA as u32);
        let n = points.len();

        // base case
        if n < SIMPLE_THRESHOLD {
            return Self::build_simple(points, None, bbox);
        }

        // oversample 2^lambda * sigma elements
        let sampled = (0..SAMPLE_SIZE * SIGMA)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng();
                points[rng.random_range(0..points.len())]
            })
            .collect::<Vec<_>>();

        // build skeleton
        let skeleton = Skeleton::build(sampled, bbox);
        assert!(
            skeleton.num_buckets() == SAMPLE_SIZE,
            "when building from scratch, the skeleton must be balanced"
        );

        // partition all points into buckets, i.e. leaves of the skeleton
        let (_, bucket_sizes) = points.sieve(SAMPLE_SIZE, &|p| skeleton.lookup(*p) as u8);

        // bucket size tree
        let size_skel = skeleton.size_tree(&bucket_sizes);
        Self::build_helper(points, &size_skel.flattened, 1, bbox)
    }
}

impl<C: Coord, const K: usize> From<&Node<C, K>> for Skeleton<C> {
    fn from(node: &Node<C, K>) -> Skeleton<C> {
        // flatten the skeleton to speed up lookups
        fn recurse<C: Coord, const K: usize>(
            curr: &Node<C, K>,
            depth: usize,
            bucket_id: &mut usize,
            skel: &mut [SkeletonEntry<C>],
            skel_id: usize,
        ) {
            match curr {
                Node::Interior {
                    size,
                    splitter,
                    bbox,
                    left,
                    right,
                } if depth < LAMBDA => {
                    // store splitter into skel
                    skel[skel_id] = SkeletonEntry::Splitter {
                        total: *size,
                        split: left.size(),
                        splitter: *splitter,
                    };

                    // recursively reconstruct skeletal tree
                    recurse(left, depth + 1, bucket_id, skel, 2 * skel_id);
                    recurse(right, depth + 1, bucket_id, skel, 2 * skel_id + 1);
                }
                _ => {
                    // skeleton stores bucket if input tree has a leaf, or if depth LAMBDA hit
                    debug_assert!(depth >= LAMBDA || matches!(curr, Node::Leaf { .. }));
                    skel[skel_id] = SkeletonEntry::Bucket(*bucket_id);
                    *bucket_id = *bucket_id + 1;
                }
            }
        }
        let mut bucket_id = 0;
        let mut flattened = vec![Default::default(); 2usize.pow(1 + LAMBDA as u32)];
        recurse(node, 0, &mut bucket_id, &mut flattened, 1);
        Skeleton {
            n_buckets: bucket_id,
            flattened,
        }
    }
}

impl<C: Coord> Skeleton<C> {
    // builds skeleton from sample
    pub fn build<const K: usize>(mut points: Vec<[C; K]>, bbox: BBox<C, K>) -> Skeleton<C> {
        // build tree on points
        let skel_tree = Node::build_simple(&mut points, Some(LAMBDA), bbox);
        Self::from(&skel_tree)
    }

    // bucket count
    pub fn num_buckets(&self) -> usize {
        self.n_buckets
    }

    // for a given point, find which bucket (leaf) of the skeleton it goes in
    pub fn lookup<const K: usize>(&self, point: [C; K]) -> usize {
        // standard PKD point search, but on this heap tree format instead
        let mut skel_id = 1;
        while skel_id < self.flattened.len() {
            match &self.flattened[skel_id] {
                SkeletonEntry::Bucket(id) => return *id,
                SkeletonEntry::Splitter {
                    splitter: (split_val, split_dim),
                    ..
                } => {
                    skel_id = skel_id * 2 + (point[*split_dim] >= *split_val) as usize;
                }
                SkeletonEntry::Blank => unreachable!("illegal"),
            }
        }
        unreachable!("bad skeleton")
    }

    pub fn size_tree(&self, bucket_sizes: &[usize]) -> Skeleton<C> {
        fn recurse<C: Coord>(
            skel: &[SkeletonEntry<C>],
            size_skel: &mut [SkeletonEntry<C>],
            skel_id: usize,
            bucket_sizes: &[usize],
        ) -> usize {
            match &skel[skel_id] {
                SkeletonEntry::Bucket(id) => {
                    let size = bucket_sizes[*id];
                    size_skel[skel_id] = SkeletonEntry::Bucket(*id);
                    size
                }
                SkeletonEntry::Splitter { splitter, .. } => {
                    let (new_left_size, new_right_size) = (
                        recurse(skel, size_skel, 2 * skel_id, bucket_sizes),
                        recurse(skel, size_skel, 2 * skel_id + 1, bucket_sizes),
                    );
                    let total = new_left_size + new_right_size;
                    size_skel[skel_id] = SkeletonEntry::Splitter {
                        total,
                        split: new_left_size,
                        splitter: *splitter,
                    };
                    total
                }
                SkeletonEntry::Blank => unreachable!("illegal"),
            }
        }
        let mut size_skel = vec![Default::default(); self.flattened.len()];
        recurse(&self.flattened, &mut size_skel, 1, bucket_sizes);
        Skeleton {
            n_buckets: self.n_buckets,
            flattened: size_skel,
        }
    }
}

impl<C: Coord, const K: usize> PKDTree<C, K> {
    pub fn build_simple(mut points: Vec<[C; K]>) -> Self {
        let bbox = BBox::build(&points);
        Self(Node::build_simple(&mut points, None, bbox))
    }
}
