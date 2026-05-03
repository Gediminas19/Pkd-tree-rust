use std::cmp::Ordering;

use rand::prelude::*;
use rayon::prelude::*;
use rayon_scan::ScanParallelIterator;

use crate::sieve::{Basic, Sieve};

pub fn kth<P: Basic>(elems: &mut [P], k: usize, cmp: &(impl Sync + Fn(&P, &P) -> Ordering)) -> P {
    // println!("Find {}-th element out of {} elements", k, elems.len());
    if elems.len() <= 1000 {
        // let mut elems = elems.to_vec();
        return *elems.select_nth_unstable_by(k, cmp).1;
    }

    // oversample sample_size*over_rate elements
    const SAMPLE_SIZE: usize = 31;
    let over_rate = 8;
    let mut sampled = (0..SAMPLE_SIZE * over_rate)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::rng();
            elems[rng.random_range(0..elems.len())]
        })
        .collect::<Vec<P>>();

    // obtain pivots by taking every 8th sampled element (after sorting)
    sampled.sort_unstable_by(cmp);
    let pivots = (0..SAMPLE_SIZE)
        .into_par_iter() // probably unneeded par because only 32 iterations
        .map(|i| sampled[i * over_rate])
        .collect::<Vec<P>>();

    // the big partition
    const BUCKET_COUNT: usize = SAMPLE_SIZE + 1;
    let (bucket_borders, _) = elems.sieve(BUCKET_COUNT, &|e| {
        pivots.partition_point(|p| cmp(p, e) != Ordering::Greater) as u8
    });

    // identify which bucket k is in and recurse
    let in_bucket_id = bucket_borders.partition_point(|&p| p <= k);
    kth(
        &mut elems[bucket_borders[in_bucket_id - 1]..bucket_borders[in_bucket_id]],
        k - bucket_borders[in_bucket_id - 1],
        cmp,
    )
}

// like kth, but does NOT partition elems into those smaller and those larger, only returns kth element
pub fn kth_only<P: Basic>(elems: &[P], k: usize, cmp: &(impl Sync + Fn(&P, &P) -> Ordering)) -> P {
    println!("Find {}-th element out of {} elements", k, elems.len());
    if elems.len() <= 1000 {
        let mut elems = elems.to_vec();
        return *elems.select_nth_unstable_by(k, cmp).1;
    }

    // oversample sample_size*over_rate elements
    const SAMPLE_SIZE: usize = 31;
    let over_rate = 8;
    let mut sampled = (0..SAMPLE_SIZE * over_rate)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::rng();
            elems[rng.random_range(0..elems.len())]
        })
        .collect::<Vec<P>>();

    // obtain pivots
    sampled.sort_unstable_by(cmp);
    let pivots = (0..SAMPLE_SIZE)
        .into_par_iter() // probably unneeded par because only 32 iterations
        .map(|i| sampled[i * over_rate])
        .collect::<Vec<P>>();

    // place all points into buckets
    let bucket_of = elems
        .par_iter()
        .map(|e| pivots.partition_point(|p| cmp(p, e) != Ordering::Greater))
        .collect::<Vec<usize>>();

    // build histogram (frequency) for each bucket
    let histogram = bucket_of
        .par_iter()
        .fold(
            || vec![0usize; SAMPLE_SIZE + 1],
            |mut counters, &bucket| {
                counters[bucket] += 1;
                counters
            },
        )
        .reduce(
            || vec![0usize; SAMPLE_SIZE + 1],
            |counter1, counter2| {
                counter1
                    .into_iter()
                    .zip(counter2.into_iter())
                    .map(|(c1, c2)| c1 + c2)
                    .collect()
            },
        );

    // get rank offsets for each bucket and find which bucket to look in
    let offsets = histogram
        .into_par_iter() // again the par is probably unnecessary
        .scan(|a, b| a + b, 0)
        .collect::<Vec<_>>();
    let in_bucket_id = offsets.partition_point(|&p| p <= k);

    // recurse on that bucket
    let inner_k = if in_bucket_id == 0 {
        k
    } else {
        k - offsets[in_bucket_id - 1]
    };
    let mut inner_elems = bucket_of
        .into_par_iter()
        .enumerate()
        .filter_map(|(i, bucket_id)| {
            if bucket_id == in_bucket_id {
                Some(elems[i])
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    kth_only(&mut inner_elems, inner_k, cmp)
}

pub fn split_multi_mut<'a, T>(slice: &'a mut [T], split_idxs: &[usize]) -> Vec<&'a mut [T]> {
    let mut mut_slices = Vec::new();
    split_idxs
        .array_windows()
        .fold(slice, |rem, &[start, end]| {
            let (head, tail) = rem.split_at_mut(end - start);
            mut_slices.push(head);
            tail
        });
    mut_slices
}
