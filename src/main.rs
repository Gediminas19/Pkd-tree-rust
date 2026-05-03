#![feature(iter_partition_in_place)]

mod pkd_tree;
mod sieve;
mod utils;

use core::num;
use std::time;

use ordered_float::OrderedFloat;
use rand::{RngExt, seq::SliceRandom};
use rayon::{prelude::*, vec};
use rayon_scan::ScanParallelIterator;
use typenum::{Integer, Unsigned};

use crate::{pkd_tree::PKDTree, sieve::Basic};

// Helper macro to time evaluating an expression (like a function call.)
macro_rules! time {
    ( $x:expr ) => {{
        let t1 = time::Instant::now();
        let result = $x;
        (result, t1.elapsed())
    }};
}

fn main() {
    // uniformly randomly generate 5D points in [100]^5 hypercube
    let gen_points = |num_points| {
        let (points, gen_time) = time!(
            (0..num_points)
                .into_par_iter()
                .map(|_| {
                    let mut rng = rand::rng();
                    [
                        OrderedFloat(100. * rng.random::<f32>()),
                        OrderedFloat(100. * rng.random::<f32>()),
                        OrderedFloat(100. * rng.random::<f32>()),
                        OrderedFloat(100. * rng.random::<f32>()),
                        OrderedFloat(100. * rng.random::<f32>()),
                    ]
                })
                .collect::<Vec<_>>()
        );
        println!("Generated {} points in {} ms", num_points, gen_time.as_millis());
        points
    };

    const N_GEN_POINTS: usize = 100000000;
    const N_INS_POINTS: usize = N_GEN_POINTS / 100;

    let points = gen_points(N_GEN_POINTS);
    let (lib_tree, lib_time) = time!(kd_tree::KdTreeN::par_build_by_ordered_float(points));
    println!(
        "Generated lib tree from {} points in {} ms",
        N_GEN_POINTS,
        lib_time.as_millis()
    );

    let points = gen_points(N_GEN_POINTS);
    let (mut fast_tree, fast_time) = time!(PKDTree::build(points));
    println!(
        "Generated fast tree from {} points in {} ms",
        N_GEN_POINTS,
        fast_time.as_millis()
    );

    let insert_points = gen_points(N_INS_POINTS);
    let ((), insert_time) = time!(fast_tree.batch_insert(insert_points));
    println!(
        "Inserted {} points into fast tree in {} ms",
        N_INS_POINTS,
        insert_time.as_millis()
    );

    // let (slow_tree, slow_time) = time!(PKDTree::build_simple(points));
    // println!("Generated slow tree in {} ms", slow_time.as_millis());
}
