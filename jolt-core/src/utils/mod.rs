use crate::field::{ChallengeFieldOps, JoltField};

use rayon::prelude::*;
use vstd::prelude::*;
use interleaving_spec::*;

pub mod accumulation;
pub mod counters;
pub mod errors;
pub mod expanding_table;
pub mod gaussian_elimination;
pub mod lookup_bits;
pub mod math;
#[cfg(feature = "monitor")]
pub mod monitor;
pub mod profiling;
pub mod small_scalar;
pub mod thread;
pub mod interleaving_spec;

/// Converts an integer value to a bitvector (all values {0,1}) of field elements.
/// Note: ordering has the MSB in the highest index. All of the following represent the integer 1:
/// - [1]
/// - [0, 0, 1]
/// - [0, 0, 0, 0, 0, 0, 0, 1]
/// ```ignore
/// use jolt_core::utils::index_to_field_bitvector;
/// # use ark_bn254::Fr;
/// # use ark_std::{One, Zero};
/// let zero = Fr::zero();
/// let one = Fr::one();
///
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 1), vec![one]);
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 3), vec![zero, zero, one]);
/// assert_eq!(index_to_field_bitvector::<Fr>(1, 7), vec![zero, zero, zero, zero, zero, zero, one]);
/// ```
pub fn index_to_field_bitvector<F: JoltField + ChallengeFieldOps<F>>(
    value: u128,
    bits: usize,
) -> Vec<F> {
    if bits != 128 {
        assert!(value < 1u128 << bits);
    }

    let mut bitvector: Vec<F> = Vec::with_capacity(bits);

    for i in (0..bits).rev() {
        if (value >> i) & 1 == 1 {
            bitvector.push(F::one());
        } else {
            bitvector.push(F::zero());
        }
    }
    bitvector
}

#[tracing::instrument(skip_all)]
pub fn compute_dotproduct<F: JoltField>(a: &[F], b: &[F]) -> F {
    a.par_iter()
        .zip_eq(b.par_iter())
        .map(|(a_i, b_i)| *a_i * *b_i)
        .sum()
}

/// Compute dotproduct optimized for values being 0 / 1
#[tracing::instrument(skip_all)]
pub fn compute_dotproduct_low_optimized<F: JoltField>(a: &[F], b: &[F]) -> F {
    a.par_iter()
        .zip_eq(b.par_iter())
        .map(|(a_i, b_i)| mul_0_1_optimized(a_i, b_i))
        .sum()
}

#[inline(always)]
pub fn mul_0_1_optimized<F: JoltField>(a: &F, b: &F) -> F {
    if a.is_zero() || b.is_zero() {
        F::zero()
    } else if a.is_one() {
        *b
    } else if b.is_one() {
        *a
    } else {
        *a * *b
    }
}

#[inline(always)]
pub fn mul_0_optimized<F: JoltField>(likely_zero: &F, x: &F) -> F {
    if likely_zero.is_zero() {
        F::zero()
    } else {
        *likely_zero * *x
    }
}

verus! {

/// Splits a 128-bit value into two 64-bit values by separating even and odd bits.
/// The even bits (indices 0,2,4,...) go into the first returned value, and odd bits (indices 1,3,5,...) into the second.
///
/// # Arguments
///
/// * `val` - The 128-bit input value to split
///
/// # Returns
///
/// A tuple (x, y) where:
/// - x contains the bits from even indices (0,2,4,...) compacted into the low 64 bits
/// - y contains the bits from odd indices (1,3,5,...) compacted into the low 64 bits
pub fn uninterleave_bits(val: u128) -> (result: (u64, u64))
    ensures result == spec_uninterleave_bits(val)
{
    // Isolate even and odd bits.
    let mut x_bits = (val >> 1) & 0x5555_5555_5555_5555_5555_5555_5555_5555;
    let mut y_bits = val & 0x5555_5555_5555_5555_5555_5555_5555_5555;
    // Compact the bits into the lower part of `x_bits`
    x_bits = (x_bits | (x_bits >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x_bits = (x_bits | (x_bits >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    x_bits = (x_bits | (x_bits >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    x_bits = (x_bits | (x_bits >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    x_bits = (x_bits | (x_bits >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    x_bits = (x_bits | (x_bits >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;
    // And do the same for `y_bits`
    y_bits = (y_bits | (y_bits >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    y_bits = (y_bits | (y_bits >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    y_bits = (y_bits | (y_bits >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    y_bits = (y_bits | (y_bits >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    y_bits = (y_bits | (y_bits >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    y_bits = (y_bits | (y_bits >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;
    
    assert(forall|j: nat| j < 64 ==> #[trigger] get_bit_u64(x_bits as u64,j as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),j as u64)) by {
        assert_forall_by(|j: nat| {
            requires(j < 64);
            ensures(#[trigger] get_bit_u64(x_bits as u64,j as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),j as u64));
            let d = (val >> 1) & 0x5555_5555_5555_5555_5555_5555_5555_5555;
            let c = (d | (d >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
            let b = (c | (c >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
            let a = (b | (b >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
            let e = (a | (a >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
            let f = (e | (e >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
            let g = (f | (f >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;
            let v0 = get_bit_u128(val,(2*j+1) as u128);
            let ap0 = get_bit_u128(x_bits,j as u128);
            let x = spec_extract_odd_bits_u128(val);
            let a0 = get_bit_u64(x,j as u64);
            spec_extract_odd_bits_u128_correctness(val, j as u128);
            let jj = j as u128;
            assert(a0 == v0);
            assert(a0 == ap0) by (bit_vector)
                requires 
                    a0 == v0,
                    v0 == (0x1u128 & (val >> (2*jj+1)) == 1),
                    d == ((val >> 1) & 0x5555_5555_5555_5555_5555_5555_5555_5555),
                    c == (d | (d >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333,
                    b == (c | (c >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F,
                    a == (b | (b >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF,
                    e == (a | (a >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF,
                    f == (e | (e >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF,
                    g == (f | (f >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF,
                    ap0 == (0x1u128 & (g >> (jj)) == 1);
            casting128_to_64(x_bits, j as u128);
        });
    }

    assert(forall|j: nat| j < 64 ==> #[trigger] get_bit_u64(y_bits as u64,j as u64) == get_bit_u64(spec_extract_even_bits_u128(val),j as u64)) by {
        assert_forall_by(|j: nat| {
            requires(j < 64);
            ensures(#[trigger] get_bit_u64(y_bits as u64,j as u64) == get_bit_u64(spec_extract_even_bits_u128(val),j as u64));
            let d = val & 0x5555_5555_5555_5555_5555_5555_5555_5555;
            let c = (d | (d >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
            let b = (c | (c >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
            let a = (b | (b >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
            let e = (a | (a >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
            let f = (e | (e >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
            let g = (f | (f >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;
            let v0 = get_bit_u128(val,(2*j) as u128);
            let ap0 = get_bit_u128(y_bits,j as u128);
            let y = spec_extract_even_bits_u128(val);
            let a0 = get_bit_u64(y,j as u64);
            spec_extract_even_bits_u128_correctness(val, j as u128);
            let jj = j as u128;
            assert(a0 == v0);
            assert(a0 == ap0) by (bit_vector)
                requires 
                    a0 == v0,
                    v0 == (0x1u128 & (val >> (2*jj)) == 1),
                    d == (val & 0x5555_5555_5555_5555_5555_5555_5555_5555),
                    c == (d | (d >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333,
                    b == (c | (c >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F,
                    a == (b | (b >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF,
                    e == (a | (a >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF,
                    f == (e | (e >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF,
                    g == (f | (f >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF,
                    ap0 == (0x1u128 & (g >> (jj)) == 1);
            casting128_to_64(y_bits, j as u128);
        });
    }
    proof{
        assert(get_bit_u64(y_bits as u64,0 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),0 as u64));
        assert(get_bit_u64(y_bits as u64,1 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),1 as u64));
        assert(get_bit_u64(y_bits as u64,2 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),2 as u64));
        assert(get_bit_u64(y_bits as u64,3 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),3 as u64));
        assert(get_bit_u64(y_bits as u64,4 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),4 as u64));
        assert(get_bit_u64(y_bits as u64,5 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),5 as u64));
        assert(get_bit_u64(y_bits as u64,6 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),6 as u64));
        assert(get_bit_u64(y_bits as u64,7 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),7 as u64));
        assert(get_bit_u64(y_bits as u64,8 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),8 as u64));
        assert(get_bit_u64(y_bits as u64,9 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),9 as u64));
        assert(get_bit_u64(y_bits as u64,10 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),10 as u64));
        assert(get_bit_u64(y_bits as u64,11 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),11 as u64));
        assert(get_bit_u64(y_bits as u64,12 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),12 as u64));
        assert(get_bit_u64(y_bits as u64,13 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),13 as u64));
        assert(get_bit_u64(y_bits as u64,14 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),14 as u64));
        assert(get_bit_u64(y_bits as u64,15 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),15 as u64));
        assert(get_bit_u64(y_bits as u64,16 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),16 as u64));
        assert(get_bit_u64(y_bits as u64,17 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),17 as u64));
        assert(get_bit_u64(y_bits as u64,18 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),18 as u64));
        assert(get_bit_u64(y_bits as u64,19 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),19 as u64));
        assert(get_bit_u64(y_bits as u64,20 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),20 as u64));
        assert(get_bit_u64(y_bits as u64,21 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),21 as u64));
        assert(get_bit_u64(y_bits as u64,22 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),22 as u64));
        assert(get_bit_u64(y_bits as u64,23 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),23 as u64));
        assert(get_bit_u64(y_bits as u64,24 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),24 as u64));
        assert(get_bit_u64(y_bits as u64,25 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),25 as u64));
        assert(get_bit_u64(y_bits as u64,26 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),26 as u64));
        assert(get_bit_u64(y_bits as u64,27 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),27 as u64));
        assert(get_bit_u64(y_bits as u64,28 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),28 as u64));
        assert(get_bit_u64(y_bits as u64,29 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),29 as u64));
        assert(get_bit_u64(y_bits as u64,30 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),30 as u64));
        assert(get_bit_u64(y_bits as u64,31 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),31 as u64));
        assert(get_bit_u64(y_bits as u64,32 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),32 as u64));
        assert(get_bit_u64(y_bits as u64,33 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),33 as u64));
        assert(get_bit_u64(y_bits as u64,34 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),34 as u64));
        assert(get_bit_u64(y_bits as u64,35 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),35 as u64));
        assert(get_bit_u64(y_bits as u64,36 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),36 as u64));
        assert(get_bit_u64(y_bits as u64,37 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),37 as u64));
        assert(get_bit_u64(y_bits as u64,38 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),38 as u64));
        assert(get_bit_u64(y_bits as u64,39 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),39 as u64));
        assert(get_bit_u64(y_bits as u64,40 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),40 as u64));
        assert(get_bit_u64(y_bits as u64,41 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),41 as u64));
        assert(get_bit_u64(y_bits as u64,42 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),42 as u64));
        assert(get_bit_u64(y_bits as u64,43 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),43 as u64));
        assert(get_bit_u64(y_bits as u64,44 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),44 as u64));
        assert(get_bit_u64(y_bits as u64,45 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),45 as u64));
        assert(get_bit_u64(y_bits as u64,46 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),46 as u64));
        assert(get_bit_u64(y_bits as u64,47 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),47 as u64));
        assert(get_bit_u64(y_bits as u64,48 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),48 as u64));
        assert(get_bit_u64(y_bits as u64,49 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),49 as u64));
        assert(get_bit_u64(y_bits as u64,50 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),50 as u64));
        assert(get_bit_u64(y_bits as u64,51 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),51 as u64));
        assert(get_bit_u64(y_bits as u64,52 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),52 as u64));
        assert(get_bit_u64(y_bits as u64,53 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),53 as u64));
        assert(get_bit_u64(y_bits as u64,54 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),54 as u64));
        assert(get_bit_u64(y_bits as u64,55 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),55 as u64));
        assert(get_bit_u64(y_bits as u64,56 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),56 as u64));
        assert(get_bit_u64(y_bits as u64,57 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),57 as u64));
        assert(get_bit_u64(y_bits as u64,58 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),58 as u64));
        assert(get_bit_u64(y_bits as u64,59 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),59 as u64));
        assert(get_bit_u64(y_bits as u64,60 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),60 as u64));
        assert(get_bit_u64(y_bits as u64,61 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),61 as u64));
        assert(get_bit_u64(y_bits as u64,62 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),62 as u64));
        assert(get_bit_u64(y_bits as u64,63 as u64) == get_bit_u64(spec_extract_even_bits_u128(val),63 as u64));
        equality_of_vals_u64(y_bits as u64, spec_extract_even_bits_u128(val));

        assert(get_bit_u64(x_bits as u64,0 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),0 as u64));
        assert(get_bit_u64(x_bits as u64,1 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),1 as u64));
        assert(get_bit_u64(x_bits as u64,2 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),2 as u64));
        assert(get_bit_u64(x_bits as u64,3 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),3 as u64));
        assert(get_bit_u64(x_bits as u64,4 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),4 as u64));
        assert(get_bit_u64(x_bits as u64,5 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),5 as u64));
        assert(get_bit_u64(x_bits as u64,6 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),6 as u64));
        assert(get_bit_u64(x_bits as u64,7 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),7 as u64));
        assert(get_bit_u64(x_bits as u64,8 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),8 as u64));
        assert(get_bit_u64(x_bits as u64,9 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),9 as u64));
        assert(get_bit_u64(x_bits as u64,10 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),10 as u64));
        assert(get_bit_u64(x_bits as u64,11 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),11 as u64));
        assert(get_bit_u64(x_bits as u64,12 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),12 as u64));
        assert(get_bit_u64(x_bits as u64,13 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),13 as u64));
        assert(get_bit_u64(x_bits as u64,14 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),14 as u64));
        assert(get_bit_u64(x_bits as u64,15 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),15 as u64));
        assert(get_bit_u64(x_bits as u64,16 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),16 as u64));
        assert(get_bit_u64(x_bits as u64,17 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),17 as u64));
        assert(get_bit_u64(x_bits as u64,18 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),18 as u64));
        assert(get_bit_u64(x_bits as u64,19 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),19 as u64));
        assert(get_bit_u64(x_bits as u64,20 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),20 as u64));
        assert(get_bit_u64(x_bits as u64,21 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),21 as u64));
        assert(get_bit_u64(x_bits as u64,22 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),22 as u64));
        assert(get_bit_u64(x_bits as u64,23 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),23 as u64));
        assert(get_bit_u64(x_bits as u64,24 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),24 as u64));
        assert(get_bit_u64(x_bits as u64,25 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),25 as u64));
        assert(get_bit_u64(x_bits as u64,26 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),26 as u64));
        assert(get_bit_u64(x_bits as u64,27 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),27 as u64));
        assert(get_bit_u64(x_bits as u64,28 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),28 as u64));
        assert(get_bit_u64(x_bits as u64,29 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),29 as u64));
        assert(get_bit_u64(x_bits as u64,30 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),30 as u64));
        assert(get_bit_u64(x_bits as u64,31 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),31 as u64));
        assert(get_bit_u64(x_bits as u64,32 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),32 as u64));
        assert(get_bit_u64(x_bits as u64,33 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),33 as u64));
        assert(get_bit_u64(x_bits as u64,34 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),34 as u64));
        assert(get_bit_u64(x_bits as u64,35 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),35 as u64));
        assert(get_bit_u64(x_bits as u64,36 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),36 as u64));
        assert(get_bit_u64(x_bits as u64,37 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),37 as u64));
        assert(get_bit_u64(x_bits as u64,38 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),38 as u64));
        assert(get_bit_u64(x_bits as u64,39 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),39 as u64));
        assert(get_bit_u64(x_bits as u64,40 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),40 as u64));
        assert(get_bit_u64(x_bits as u64,41 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),41 as u64));
        assert(get_bit_u64(x_bits as u64,42 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),42 as u64));
        assert(get_bit_u64(x_bits as u64,43 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),43 as u64));
        assert(get_bit_u64(x_bits as u64,44 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),44 as u64));
        assert(get_bit_u64(x_bits as u64,45 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),45 as u64));
        assert(get_bit_u64(x_bits as u64,46 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),46 as u64));
        assert(get_bit_u64(x_bits as u64,47 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),47 as u64));
        assert(get_bit_u64(x_bits as u64,48 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),48 as u64));
        assert(get_bit_u64(x_bits as u64,49 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),49 as u64));
        assert(get_bit_u64(x_bits as u64,50 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),50 as u64));
        assert(get_bit_u64(x_bits as u64,51 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),51 as u64));
        assert(get_bit_u64(x_bits as u64,52 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),52 as u64));
        assert(get_bit_u64(x_bits as u64,53 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),53 as u64));
        assert(get_bit_u64(x_bits as u64,54 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),54 as u64));
        assert(get_bit_u64(x_bits as u64,55 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),55 as u64));
        assert(get_bit_u64(x_bits as u64,56 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),56 as u64));
        assert(get_bit_u64(x_bits as u64,57 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),57 as u64));
        assert(get_bit_u64(x_bits as u64,58 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),58 as u64));
        assert(get_bit_u64(x_bits as u64,59 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),59 as u64));
        assert(get_bit_u64(x_bits as u64,60 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),60 as u64));
        assert(get_bit_u64(x_bits as u64,61 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),61 as u64));
        assert(get_bit_u64(x_bits as u64,62 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),62 as u64));
        assert(get_bit_u64(x_bits as u64,63 as u64) == get_bit_u64(spec_extract_odd_bits_u128(val),63 as u64));
        equality_of_vals_u64(x_bits as u64, spec_extract_odd_bits_u128(val));
    }
    (x_bits as u64, y_bits as u64)

}

}

/// Combines two 64-bit values into a single 128-bit value by interleaving their bits.
/// Takes even bits from the first argument and odd bits from the second argument.
///
/// # Arguments
///
/// * `even_bits` - A 64-bit value whose bits will be placed at even indices (0,2,4,...)
/// * `odd_bits` - A 64-bit value whose bits will be placed at odd indices (1,3,5,...)
///
/// # Returns
///
/// A 128-bit value containing interleaved bits from the input values, with even_bits shifted into even positions
/// and odd_bits in odd positions.
///
/// # Examples
///
/// ```
/// # use jolt_core::utils::interleave_bits;
/// assert_eq!(interleave_bits(0b01, 0b10), 0b110);
/// ```
pub fn interleave_bits(even_bits: u64, odd_bits: u64) -> u128 {
    // Insert zeros between each bit of `x_bits`
    let mut x_bits = even_bits as u128;
    x_bits = (x_bits | (x_bits << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    x_bits = (x_bits | (x_bits << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    x_bits = (x_bits | (x_bits << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    x_bits = (x_bits | (x_bits << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    x_bits = (x_bits | (x_bits << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x_bits = (x_bits | (x_bits << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    // And do the same for `y_bits`
    let mut y_bits = odd_bits as u128;
    y_bits = (y_bits | (y_bits << 32)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    y_bits = (y_bits | (y_bits << 16)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    y_bits = (y_bits | (y_bits << 8)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    y_bits = (y_bits | (y_bits << 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    y_bits = (y_bits | (y_bits << 2)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    y_bits = (y_bits | (y_bits << 1)) & 0x5555_5555_5555_5555_5555_5555_5555_5555;

    (x_bits << 1) | y_bits
}

#[cfg(test)]
mod tests {
    use ark_std::test_rng;
    use rand_core::RngCore;

    use super::*;

    #[test]
    fn interleave_uninterleave_bits() {
        let mut rng = test_rng();
        for _ in 0..1000 {
            let val = ((rng.next_u64() as u128) << 64) | rng.next_u64() as u128;
            let (even, odd) = uninterleave_bits(val);
            assert_eq!(val, interleave_bits(even, odd));
        }

        for _ in 0..1000 {
            let even = rng.next_u64();
            let odd = rng.next_u64();
            assert_eq!((even, odd), uninterleave_bits(interleave_bits(even, odd)));
        }
    }
}
