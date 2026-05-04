use crate::{
    pkd_tree::{ALPHA, BBox, Coord, Node, PHI, PKDTree, SIMPLE_THRESHOLD, Skeleton, SkeletonEntry},
    sieve::Sieve,
};

impl<C: Coord, const K: usize> Node<C, K> {
    // consume subtree in parallel, get all the points out
    fn flatten(&self, dest: &mut [[C; K]]) {
        match self {
            Node::Leaf { points, .. } => dest.copy_from_slice(&points),
            Node::Interior { left, right, .. } => {
                let (left_dest, right_dest) = dest.split_at_mut(left.size());
                rayon::join(|| left.flatten(left_dest), || right.flatten(right_dest));
            }
        }
    }

    fn batch_insert_simple(&mut self, new_points: &mut [[C; K]], bbox: BBox<C, K>) {
        match self {
            Node::Leaf { bbox, points } => {
                points.extend_from_slice(new_points);
                // rebuild leaf node if too many points
                if points.len() > PHI {
                    *self = Self::build_simple(points, None, *bbox);
                }
            }
            _ => {
                let total = new_points.len();

                let size = if let Node::Interior {
                    left,
                    right,
                    splitter: (median, axis),
                    size,
                    ..
                } = self
                {
                    let split = new_points.iter_mut().partition_in_place(|p| p[*axis] < *median);

                    // get the total # of points that will go to left vs right subtree AFTER insert
                    if f32::abs(((split + left.size()) as f32 / (total + *size) as f32) - 0.5) <= ALPHA {
                        // split new points between left and right
                        let (left_points, right_points) = new_points.split_at_mut(split);

                        // split bbox
                        let (lbbox, rbbox) = bbox.split(*median, *axis);

                        // insert into subtrees in parallel
                        rayon::join(
                            || left.batch_insert_simple(left_points, lbbox),
                            || right.batch_insert_simple(right_points, rbbox),
                        );
                        return;
                    } else {
                        *size
                    }
                } else {
                    unreachable!("no other cases")
                };

                // collect all the points to use during rebuilding
                let new_total = total + size;
                let mut all_points = Vec::with_capacity(new_total);
                unsafe {
                    all_points.set_len(new_total);
                }
                self.flatten(&mut all_points[0..size]);
                all_points[size..new_total].copy_from_slice(new_points);

                // rebuild with old and new points
                *self = Self::build(&mut all_points, bbox);
            }
        }
    }

    fn batch_insert_helper(
        &mut self,
        new_points: &mut [[C; K]],
        size_skel: &[SkeletonEntry<C>],
        skel_id: usize,
        bbox: BBox<C, K>,
    ) {
        match &size_skel[skel_id] {
            SkeletonEntry::Bucket { .. } => self.batch_insert(new_points, bbox),
            SkeletonEntry::Splitter { total, split, .. } => {
                // check if resulting subtrees would be imbalanced
                let size = if let Node::Interior {
                    left,
                    right,
                    splitter,
                    size,
                    ..
                } = self
                {
                    // get the total # of points that will go to left vs right subtree AFTER insert
                    if f32::abs(((split + left.size()) as f32 / (*total + *size) as f32) - 0.5) <= ALPHA {
                        // split new points between left and right
                        let (left_points, right_points) = new_points.split_at_mut(*split);

                        // split bbox
                        let (median, axis) = splitter;
                        let (lbbox, rbbox) = bbox.split(*median, *axis);

                        // insert into subtrees in parallel
                        rayon::join(
                            || left.batch_insert_helper(left_points, size_skel, 2 * skel_id, lbbox),
                            || right.batch_insert_helper(right_points, size_skel, 2 * skel_id + 1, rbbox),
                        );
                        return;
                    } else {
                        *size
                    }
                } else {
                    unreachable!("tree and skeleton mismatch!")
                };

                // collect all the points to use during rebuilding
                let new_total = *total + size;
                let mut all_points = Vec::with_capacity(new_total);
                unsafe {
                    all_points.set_len(new_total);
                }
                self.flatten(&mut all_points[0..size]);
                all_points[size..new_total].copy_from_slice(new_points);

                // rebuild with old and new points
                *self = Self::build(&mut all_points, bbox);
            }
            SkeletonEntry::Blank => unreachable!("illegal"),
        }
    }

    pub(crate) fn batch_insert(&mut self, new_points: &mut [[C; K]], bbox: BBox<C, K>) {
        match self {
            // base case
            Node::Leaf { points, bbox } => {
                points.extend_from_slice(new_points);
                // rebuild leaf node if too many points
                if points.len() > PHI {
                    *self = Self::build(points, *bbox);
                }
            }

            Node::Interior { size, .. } => {
                // base case
                if *size < SIMPLE_THRESHOLD {
                    return self.batch_insert_simple(new_points, bbox);
                }

                // get skeleton from tree itself
                let skeleton = Skeleton::from(&*self);

                // partition all points into buckets, i.e. leaves of the skeleton
                let (_, bucket_sizes) = new_points.sieve(skeleton.num_buckets(), &|p| skeleton.lookup(*p) as u8);

                // bucket size tree
                let size_skel = skeleton.size_tree(&bucket_sizes);
                self.batch_insert_helper(new_points, &size_skel.flattened, 1, bbox);
            }
        }
    }
}

impl<C: Coord, const K: usize> PKDTree<C, K> {
    pub fn batch_insert_simple(&mut self, mut points: Vec<[C; K]>) {
        let bbox = BBox::merge(BBox::build(&points), self.bbox());
        self.0.batch_insert_simple(&mut points, bbox);
    }
}
