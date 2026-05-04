#![feature(iter_partition_in_place)]

mod batch_par_kd_tree_trait;
mod bbox;
mod lib_kd_tree;
mod pkd_tree;
mod sieve;
mod utils;

use std::io::BufRead;
use std::path::Path;
use std::time;

use ordered_float::OrderedFloat;
use rand::RngExt;
use rayon::prelude::*;

use crate::{
    batch_par_kd_tree_trait::{BatchParKDTree, Coord},
    lib_kd_tree::{LibKDTree2, LibKDTree3, LibKDTree5},
    pkd_tree::{PKDTree, SimplePKDTree},
};

// Helper macro to time evaluating an expression (like a function call.)
macro_rules! time {
    ( $x:expr ) => {{
        let t1 = time::Instant::now();
        let result = $x;
        (result, t1.elapsed())
    }};
}

/// Parses a file containing points into a Vec<[u32; D]>.
pub fn parse_points<const D: usize, P: AsRef<Path>>(
    filepath: P,
) -> Result<Vec<[OrderedFloat<f32>; D]>, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(filepath)?;
    let (_, points_str) = contents.split_once('\n').unwrap();

    Ok(points_str
        .par_split('\n')
        .filter_map(|line| {
            line.split(' ')
                .filter_map(|num| num.parse::<OrderedFloat<f32>>().ok())
                .collect::<Vec<_>>()
                .try_into()
                .ok()
        })
        .collect())
}

// uniformly randomly generate points in [1000]^5 hypercube
pub fn gen_uniform_points<const D: usize>(num_points: usize) -> Vec<[OrderedFloat<f32>; D]> {
    let (points, gen_time) = time!(
        (0..num_points)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng();
                std::array::from_fn(|_| OrderedFloat(500000. * rng.random::<f32>()))
            })
            .collect::<Vec<_>>()
    );
    // println!("Generated {} points in {} ms", num_points, gen_time.as_millis());
    points
}

pub trait BatchParKDTreeBench<C: Coord, const K: usize> {
    fn test_tree(points: Vec<[C; K]>) -> (u128, u128, u128);
}

// 2. Provide a blanket implementation for ANY type `T`
// that implements `BatchParKDTree<C, K>`.
impl<const K: usize, T: BatchParKDTree<OrderedFloat<f32>, K>> BatchParKDTreeBench<OrderedFloat<f32>, K> for T {
    fn test_tree(points: Vec<[OrderedFloat<f32>; K]>) -> (u128, u128, u128) {
        // generate points for querying and insertion
        let n = points.len();
        let query_points = gen_uniform_points::<K>(n / 100);
        let insert_points = gen_uniform_points::<K>(n / 10);

        let (mut tree, gen_time) = time!(Self::build(points));
        eprintln!("generated tree with {} points in {} ms", n, gen_time.as_millis());
        let (_, knn_time) = time!(tree.batch_nearests(&query_points, 10));
        eprintln!(
            "queried {} points in tree with {} points in {} ms",
            n / 100,
            n,
            knn_time.as_millis()
        );
        let (_, insert_time) = time!(tree.batch_insert(insert_points));
        eprintln!(
            "inserted {} points in tree with {} points in {} ms",
            n / 10,
            n,
            insert_time.as_millis()
        );
        (gen_time.as_millis(), knn_time.as_millis(), insert_time.as_millis())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // println!("treetype,distribution,dimension,build_time,nn_time,insert_time");
    const DIM: usize = 5;

    for i in 1..=5 {
        // let points = gen_points(N_GEN_POINTS);
        let path = format!("data/uniform/100000000_{}/{}.in", DIM, i);
        let points = parse_points::<DIM, _>(path)?;

        let (gen_time, knn_time, insert_time) = LibKDTree5::test_tree(points);
        println!("lib,{},{},{},{},{}", "uniform", DIM, gen_time, knn_time, insert_time);
    }
    for i in 1..=5 {
        // let points = gen_points(N_GEN_POINTS);
        let path = format!("data/uniform/100000000_{}/{}.in", DIM, i);
        let points = parse_points::<DIM, _>(path)?;

        let (gen_time, knn_time, insert_time) = PKDTree::test_tree(points);
        println!("pkd,{},{},{},{},{}", "uniform", DIM, gen_time, knn_time, insert_time);
    }
    for i in 1..=5 {
        // let points = gen_points(N_GEN_POINTS);
        let path = format!("data/uniform/100000000_{}/{}.in", DIM, i);
        let points = parse_points::<DIM, _>(path)?;

        let (gen_time, knn_time, insert_time) = SimplePKDTree::test_tree(points);
        println!("pnaive,{},{},{},{},{}", "uniform", DIM, gen_time, knn_time, insert_time);
    }

    Ok(())
}
