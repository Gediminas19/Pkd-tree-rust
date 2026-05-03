use std::{array, fmt::Debug};

use ordered_float::OrderedFloat;
use rayon::{iter, prelude::*};
use rayon_scan::ScanParallelIterator;
use rdst::{RadixKey, RadixSort};

pub trait Basic: Debug + Copy + Send + Sync + Ord {}

impl Basic for i32 {}
impl Basic for OrderedFloat<f32> {}

#[derive(Copy, Clone)]
struct UnsafeSlicePtr<T>(*mut T);
unsafe impl<T> Send for UnsafeSlicePtr<T> {}
unsafe impl<T> Sync for UnsafeSlicePtr<T> {}

// given a sequence of elems, and buckets that they go in, return
// - the sequence sorted by bucket (counting/bucket sort)
// - the indices demarcating the bucket borders in the sorted sequence
pub fn my_sieve<P: Basic, const B: usize>(
    elems: &[P],
    get_bucket: &(impl Sync + Fn(usize, &P) -> u8),
) -> (Vec<usize>, Vec<P>) {
    // for each "chunk", compute bucket histogram (matrix A)
    let chunk_histogram = elems
        .par_iter()
        .enumerate()
        .fold(
            || ([0usize; B], 0usize),
            |(mut counters, total), (i, elem)| {
                counters[get_bucket(i, elem) as usize] += 1;
                (counters, total + 1)
            },
        )
        .collect::<Vec<_>>();
    let num_chunks = chunk_histogram.len();
    println!("Wow {} chunks", num_chunks);

    // chunk_bucket_offsets[i][j] = starting offset within bucket j for chunk i
    let chunk_bucket_offsets = iter::once(([0usize; B], 0))
        .chain(chunk_histogram.into_par_iter())
        .scan(
            |(c1, t1), (c2, t2)| (array::from_fn(|i| c1[i] + c2[i]), t1 + t2),
            ([0usize; B], 0),
        )
        .collect::<Vec<_>>();

    // get overall offset of each bucket
    let bucket_offsets = iter::once(0)
        .chain(
            chunk_bucket_offsets[num_chunks]
                .0
                .clone() // small clone
                .into_par_iter(),
        ) // again the par is probably unnecessary
        .scan(|a, b| a + b, 0)
        .collect::<Vec<_>>();

    let mut new_elems: Vec<P> = Vec::with_capacity(elems.len());
    unsafe {
        new_elems.set_len(elems.len());
    }
    let dest_ptr = UnsafeSlicePtr(new_elems.as_mut_ptr());

    chunk_bucket_offsets
        .array_windows()
        .for_each(|[(offsets, elem_start), (_, elems_end)]| {
            let _ = (*elem_start..*elems_end).fold(offsets.clone(), |mut curr_offsets, i| {
                let bucket = get_bucket(i, &elems[i]) as usize;
                let final_offset = bucket_offsets[bucket] + curr_offsets[bucket];
                unsafe {
                    dest_ptr.0.add(final_offset).write(elems[i]);
                }
                curr_offsets[bucket] += 1;
                curr_offsets
            });
        });

    (bucket_offsets, new_elems)
}

#[derive(Copy, Clone)]
struct WithBucket<P>(P, u8);

impl<P> RadixKey for WithBucket<P> {
    const LEVELS: usize = 1;

    #[inline]
    fn get_level(&self, _level: usize) -> u8 {
        self.1
    }
}

pub trait Sieve<P: Basic> {
    fn sieve(
        &mut self,
        num_buckets: usize,
        get_bucket: &(impl Sync + Fn(&P) -> u8),
    ) -> (Vec<usize>, Vec<usize>);
}

impl<P: Basic> Sieve<P> for [P] {
    fn sieve(
        &mut self,
        num_buckets: usize,
        get_bucket: &(impl Sync + Fn(&P) -> u8),
    ) -> (Vec<usize>, Vec<usize>) {
        // tag elements with bucket, then do counting sort (1-layered radix sort)
        let mut tagged_elems = self
            .par_iter()
            .map(|p| WithBucket(*p, get_bucket(p)))
            .collect::<Vec<_>>();
        tagged_elems.radix_sort_unstable();
        let sorted_elems = tagged_elems.par_iter().map(|bp| bp.0).collect::<Vec<_>>();
        self.copy_from_slice(&sorted_elems);

        // find bucket boundaries and sizes
        let bucket_offsets = (0..=num_buckets)
            .map(|id| tagged_elems.partition_point(|bp| (bp.1 as usize) < id))
            .collect::<Vec<_>>();
        let bucket_sizes = bucket_offsets
            .array_windows()
            .map(|[start, end]| end - start)
            .collect();
        (bucket_offsets, bucket_sizes)
    }
}
