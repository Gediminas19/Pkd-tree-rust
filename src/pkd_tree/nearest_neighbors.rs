use std::collections::BinaryHeap;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::pkd_tree::{Coord, Node, PKDTree};

impl<C: Coord, const K: usize> Node<C, K> {
    pub fn nearests(&self, point: &[C; K], num: usize, heap: &mut BinaryHeap<(C, [C; K])>) {
        // squared distance between 2 points
        // TODO: there are various optimization hacks that can avoid having to compute the full sqdist
        let dist2 =
            |p1: &[C], p2: &[C]| -> C { p1.iter().zip(p2.iter()).map(|(&x1, &x2)| (x1 - x2) * (x1 - x2)).sum() };

        match self {
            Node::Leaf { points, .. } => {
                // for each leaf point, add it if the heap isn't full yet, or if it's closer than the furthest point in the heap
                points.iter().for_each(|p| {
                    let sqdist = dist2(point, p);
                    if heap.len() < num {
                        heap.push((sqdist, *p));
                    } else if sqdist < heap.peek().unwrap().0 {
                        heap.push((sqdist, *p));
                        heap.pop();
                    }
                });
            }

            Node::Interior {
                splitter: (median, axis),
                left,
                right,
                ..
            } => {
                let (look_first, look_next, dist2split) = if point[*axis] < *median {
                    (left, right, *median - point[*axis])
                } else {
                    (right, left, point[*axis] - *median)
                };

                // first search in the subtree containing the query point
                look_first.nearests(point, num, heap);

                // stop early if all num points found are closer than the splitter
                if heap.len() >= num && heap.peek().unwrap().0 < dist2split * dist2split {
                    return;
                }

                // otherwise continue searching in other subtree
                look_next.nearests(point, num, heap);
            }
        }
    }
}

impl<C: Coord, const K: usize> PKDTree<C, K> {
    pub fn batch_nearests(&self, points: &[[C; K]], num: usize) -> Vec<Vec<(C, [C; K])>> {
        points.into_par_iter().map(|p| self.nearests(p, num)).collect()
    }

    pub fn nearests(&self, point: &[C; K], num: usize) -> Vec<(C, [C; K])> {
        let mut heap = BinaryHeap::with_capacity(num);
        self.0.nearests(point, num, &mut heap);
        heap.into_sorted_vec()
    }
}
