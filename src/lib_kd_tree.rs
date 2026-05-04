use rayon::prelude::*;

use crate::batch_par_kd_tree_trait::{BatchParKDTree, Coord};

macro_rules! gen_batch_kdtree {
    ($struct_name:ident, $dim:expr) => {
        #[derive(Debug, Clone)]
        pub struct $struct_name<C: Coord>(kd_tree::KdTree<[C; $dim]>);

        impl<C: Coord> BatchParKDTree<C, $dim> for $struct_name<C> {
            fn build(points: Vec<[C; $dim]>) -> Self {
                // Use the full path or ensure KdTree is in scope
                Self(kd_tree::KdTree::par_build(points))
            }

            fn batch_insert(&mut self, mut new_points: Vec<[C; $dim]>) {
                let mut points = self.0.to_vec();
                points.append(&mut new_points);
                *self = Self(kd_tree::KdTree::par_build(points));
            }

            fn batch_nearests(&self, points: &[[C; $dim]], num: usize) -> Vec<Vec<(C, [C; $dim])>> {
                points
                    .into_par_iter()
                    .map(|p| {
                        self.0
                            .nearests(p, num)
                            .iter()
                            .map(|info| (info.squared_distance, *info.item))
                            .collect()
                    })
                    .collect()
            }
        }
    };
}

gen_batch_kdtree!(LibKDTree2, 2);
gen_batch_kdtree!(LibKDTree3, 3);
gen_batch_kdtree!(LibKDTree5, 5);
gen_batch_kdtree!(LibKDTree9, 9);
