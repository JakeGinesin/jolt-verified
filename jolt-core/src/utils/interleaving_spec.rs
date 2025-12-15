use vstd::prelude::*;
use vstd::arithmetic::*;
use vstd::arithmetic::power2::pow2;

verus! {

pub open spec fn get_bit_u128(val: u128, i: u128) -> bool
{
    0x1u128 & (val >> i) == 1
}

pub open spec fn get_bit_u64(val: u64, i: u64) -> bool
{
    0x1u64 & (val >> i) == 1
}

pub proof fn lemma_get_bit_u64_correctness(val: u64, i: u64, b: bool)
    requires
        i < 64
    ensures
        get_bit_u64(val, i) == b <==> (0x1u64 & (val >> i) == 1) == b
{
    assert(get_bit_u64(val, i) == b ==> (0x1u64 & (val >> i) == 1) == b) by {
        assume(get_bit_u64(val, i) == b);
        let bit = get_bit_u64(val, i);
        assert(bit == b) by (bit_vector)
            requires
                i < 64,
                bit == (0x1u64 & (val >> i) == 1),
                b == (0x1u64 & (val >> i) == 1);
    };

    assert((0x1u64 & (val >> i) == 1) == b ==> get_bit_u64(val, i) == b) by {
        assume((0x1u64 & (val >> i) == 1) == b);
        let bit = get_bit_u64(val, i);
        assert(bit == b) by (bit_vector)
            requires
                i < 64,
                bit == (0x1u64 & (val >> i) == 1),
                b == (0x1u64 & (val >> i) == 1);
    };
}

pub proof fn lemma_get_bit_u64_equiv(a: u64, b: u64, i: u64)
    requires
        (0x1u64 & (a >> i) == 1) == (0x1u64 & (b >> i) == 1)
    ensures 
        get_bit_u64(a, i) == get_bit_u64(b, i) && get_bit_u64(b, i) == get_bit_u64(a, i)
{
    let bit_a = get_bit_u64(a, i);
    let bit_b = get_bit_u64(b, i);
    assert(bit_a == bit_b && bit_b == bit_a) by (bit_vector)
        requires
            bit_a == (0x1u64 & (a >> i) == 1),
            bit_b == (0x1u64 & (b >> i) == 1),
            (0x1u64 & (a >> i) == 1) == (0x1u64 & (b >> i) == 1)
}

pub open spec fn build_u64_from_bits(f: spec_fn(nat) -> bool) -> u64 {
    build_u64_from_bits_rec(f, 0, 0u64)
}

pub open spec fn build_u64_from_bits_rec(f: spec_fn(nat) -> bool, i: nat, acc: u64) -> u64
    decreases 64 - i
{
    if i >= 64 {
        acc
    } else {
        let b: u64 = if f(i) { 1 } else { 0 };
        let acc2 = acc | (b << i);
        build_u64_from_bits_rec(f, i + 1, acc2)
    }
}

pub proof fn lemma_build_u64_positions_rec(
    f: spec_fn(nat) -> bool,
    i: nat,
    acc: u64,
)
    requires
        i <= 64,
        forall|j: nat| j < i ==> #[trigger] get_bit_u64(acc, j as u64) == f(j),
        forall|j: nat| i <= j && j < 64 ==> #[trigger] get_bit_u64(acc, j as u64) == false,
    ensures
        forall|j: nat| j < 64 ==> #[trigger] get_bit_u64(build_u64_from_bits_rec(f, i, acc), j as u64) == f(j),
    decreases 64 - i
{
    if i == 64 {
        assert (forall|j: nat| j < 64 ==> #[trigger] get_bit_u64(build_u64_from_bits_rec(f, i, acc), j as u64) == f(j));
    } else {
        let b: u64 = if f(i) { 1 } else { 0 };
        let acc2 = acc | (b << i);
        assert(build_u64_from_bits_rec(f, i, acc) == build_u64_from_bits_rec(f, i + 1, acc2));
        assert_forall_by(|j: nat| {
            requires(j < i+1);
            ensures(#[trigger] get_bit_u64(acc2, j as u64) == f(j));
            let jj = j as u64;
            let ii = i as u64;
            if (jj < ii) {
                assert((0x1u64 & (acc2 >> jj) == 1) == (0x1u64 & (acc >> jj) == 1)) by (bit_vector)
                requires
                    b == 0u64 || b == 1u64,
                    jj < ii,
                    jj < 64,
                    ii < 64,
                    acc2 == acc | (b << ii);
                lemma_get_bit_u64_equiv(acc2, acc, jj);
            } else {
                lemma_get_bit_u64_correctness(acc, ii, false);
                assert((0x1u64 & (acc2 >> ii) == 1) == (b == 1)) by (bit_vector)
                    requires
                        ii < 64,
                        jj == ii,
                        b == 0u64 || b == 1u64,
                        (0x1u64 & (acc >> ii) == 1) == false,
                        acc2 == acc | (b << ii);
                lemma_get_bit_u64_correctness(acc2, ii, (b == 1));
            }
        });
        assert_forall_by(|j: nat| {
            requires(j >= i+1 && j < 64);
            ensures(#[trigger] get_bit_u64(acc2, j as u64) == get_bit_u64(acc, j as u64));
            let jj = j as u64;
            let ii = i as u64;
            assert((0x1u64 & (acc2 >> jj) == 1) == (0x1u64 & (acc >> jj) == 1)) by (bit_vector)
                requires
                    b == 0u64 || b == 1u64,
                    jj >= ii + 1,
                    jj < 64,
                    ii < 64,
                    acc2 == acc | (b << ii);
            lemma_get_bit_u64_equiv(acc2, acc, jj);
        });
        lemma_build_u64_positions_rec(f, i + 1, acc2);
    }
}

pub proof fn build_u64_from_bits_correctness(f: spec_fn(nat) -> bool)
    ensures forall|i: nat| 0 <= i < 64 ==> #[trigger] get_bit_u64(build_u64_from_bits(f),i as u64) == f(i)
{
    assert_forall_by(|j: nat| {
        requires(j < 64);
        ensures(#[trigger] get_bit_u64(0u64, j as u64) == false);
        let jj = j as u64;
        assert((0x1u64 & (0u64 >> (jj)) == 1) == false) by (bit_vector);
        lemma_get_bit_u64_correctness(0u64, j as u64, false);
    });
    lemma_build_u64_positions_rec(f, 0, 0u64);
}

// Extract odd-positioned bits (1, 3, 5, ..., 127) and compact them. (Bit i of result comes from bit (2*i + 1) of input.)
pub open spec fn spec_extract_odd_bits_u128(val: u128) -> u64 
{
    build_u64_from_bits(|i: nat| 
        if i < 64 {
            get_bit_u128(val, (2 * i + 1) as u128)
        } else {
            false
        }
    )
}

// Extract even-positioned bits (0, 2, 4, ..., 126) and compact them. (Bit i of result comes from bit (2*i) of input.)
pub open spec fn spec_extract_even_bits_u128(val: u128) -> u64 
{
    build_u64_from_bits(|i: nat| 
        if i < 64 {
            get_bit_u128(val, (2 * i) as u128)
        } else {
            false
        }
    )
}

pub proof fn spec_extract_even_bits_u128_correctness(val: u128, i: u128)
    requires i < 64
    ensures get_bit_u64(spec_extract_even_bits_u128(val), i as u64) == get_bit_u128(val, (2 * i) as u128)
{
    build_u64_from_bits_correctness(|i: nat| 
        if i < 64 {
            get_bit_u128(val, (2 * i) as u128)
        } else {
            false
        });
}

pub proof fn spec_extract_odd_bits_u128_correctness(val: u128, i: u128)
    requires i < 64
    ensures get_bit_u64(spec_extract_odd_bits_u128(val), i as u64) == get_bit_u128(val, (2 * i + 1) as u128)
{
    build_u64_from_bits_correctness(|i: nat| 
        if i < 64 {
            get_bit_u128(val, (2 * i + 1) as u128)
        } else {
            false
        });
}

pub open spec fn spec_uninterleave_bits(val: u128) -> (result: (u64, u64)) {
    (spec_extract_odd_bits_u128(val), spec_extract_even_bits_u128(val))
}

pub proof fn equality_of_vals_u64(a: u64, b: u64) 
    requires 
        get_bit_u64(a,0) == get_bit_u64(b,0),
        get_bit_u64(a,1) == get_bit_u64(b,1),
        get_bit_u64(a,2) == get_bit_u64(b,2),
        get_bit_u64(a,3) == get_bit_u64(b,3),
        get_bit_u64(a,4) == get_bit_u64(b,4),
        get_bit_u64(a,5) == get_bit_u64(b,5),
        get_bit_u64(a,6) == get_bit_u64(b,6),
        get_bit_u64(a,7) == get_bit_u64(b,7),
        get_bit_u64(a,8) == get_bit_u64(b,8),
        get_bit_u64(a,9) == get_bit_u64(b,9),
        get_bit_u64(a,10) == get_bit_u64(b,10),
        get_bit_u64(a,11) == get_bit_u64(b,11),
        get_bit_u64(a,12) == get_bit_u64(b,12),
        get_bit_u64(a,13) == get_bit_u64(b,13),
        get_bit_u64(a,14) == get_bit_u64(b,14),
        get_bit_u64(a,15) == get_bit_u64(b,15),
        get_bit_u64(a,16) == get_bit_u64(b,16),
        get_bit_u64(a,17) == get_bit_u64(b,17),
        get_bit_u64(a,18) == get_bit_u64(b,18),
        get_bit_u64(a,19) == get_bit_u64(b,19),
        get_bit_u64(a,20) == get_bit_u64(b,20),
        get_bit_u64(a,21) == get_bit_u64(b,21),
        get_bit_u64(a,22) == get_bit_u64(b,22),
        get_bit_u64(a,23) == get_bit_u64(b,23),
        get_bit_u64(a,24) == get_bit_u64(b,24),
        get_bit_u64(a,25) == get_bit_u64(b,25),
        get_bit_u64(a,26) == get_bit_u64(b,26),
        get_bit_u64(a,27) == get_bit_u64(b,27),
        get_bit_u64(a,28) == get_bit_u64(b,28),
        get_bit_u64(a,29) == get_bit_u64(b,29),
        get_bit_u64(a,30) == get_bit_u64(b,30),
        get_bit_u64(a,31) == get_bit_u64(b,31),
        get_bit_u64(a,32) == get_bit_u64(b,32),
        get_bit_u64(a,33) == get_bit_u64(b,33),
        get_bit_u64(a,34) == get_bit_u64(b,34),
        get_bit_u64(a,35) == get_bit_u64(b,35),
        get_bit_u64(a,36) == get_bit_u64(b,36),
        get_bit_u64(a,37) == get_bit_u64(b,37),
        get_bit_u64(a,38) == get_bit_u64(b,38),
        get_bit_u64(a,39) == get_bit_u64(b,39),
        get_bit_u64(a,40) == get_bit_u64(b,40),
        get_bit_u64(a,41) == get_bit_u64(b,41),
        get_bit_u64(a,42) == get_bit_u64(b,42),
        get_bit_u64(a,43) == get_bit_u64(b,43),
        get_bit_u64(a,44) == get_bit_u64(b,44),
        get_bit_u64(a,45) == get_bit_u64(b,45),
        get_bit_u64(a,46) == get_bit_u64(b,46),
        get_bit_u64(a,47) == get_bit_u64(b,47),
        get_bit_u64(a,48) == get_bit_u64(b,48),
        get_bit_u64(a,49) == get_bit_u64(b,49),
        get_bit_u64(a,50) == get_bit_u64(b,50),
        get_bit_u64(a,51) == get_bit_u64(b,51),
        get_bit_u64(a,52) == get_bit_u64(b,52),
        get_bit_u64(a,53) == get_bit_u64(b,53),
        get_bit_u64(a,54) == get_bit_u64(b,54),
        get_bit_u64(a,55) == get_bit_u64(b,55),
        get_bit_u64(a,56) == get_bit_u64(b,56),
        get_bit_u64(a,57) == get_bit_u64(b,57),
        get_bit_u64(a,58) == get_bit_u64(b,58),
        get_bit_u64(a,59) == get_bit_u64(b,59),
        get_bit_u64(a,60) == get_bit_u64(b,60),
        get_bit_u64(a,61) == get_bit_u64(b,61),
        get_bit_u64(a,62) == get_bit_u64(b,62),
        get_bit_u64(a,63) == get_bit_u64(b,63)
    ensures a == b
{
    assert(a == b) by (bit_vector)
        requires
            (0x1u64 & (a >> 0) == 1) == (0x1u64 & (b >> 0) == 1),
            (0x1u64 & (a >> 1) == 1) == (0x1u64 & (b >> 1) == 1),
            (0x1u64 & (a >> 2) == 1) == (0x1u64 & (b >> 2) == 1),
            (0x1u64 & (a >> 3) == 1) == (0x1u64 & (b >> 3) == 1),
            (0x1u64 & (a >> 4) == 1) == (0x1u64 & (b >> 4) == 1),
            (0x1u64 & (a >> 5) == 1) == (0x1u64 & (b >> 5) == 1),
            (0x1u64 & (a >> 6) == 1) == (0x1u64 & (b >> 6) == 1),
            (0x1u64 & (a >> 7) == 1) == (0x1u64 & (b >> 7) == 1),
            (0x1u64 & (a >> 8) == 1) == (0x1u64 & (b >> 8) == 1),
            (0x1u64 & (a >> 9) == 1) == (0x1u64 & (b >> 9) == 1),
            (0x1u64 & (a >> 10) == 1) == (0x1u64 & (b >> 10) == 1),
            (0x1u64 & (a >> 11) == 1) == (0x1u64 & (b >> 11) == 1),
            (0x1u64 & (a >> 12) == 1) == (0x1u64 & (b >> 12) == 1),
            (0x1u64 & (a >> 13) == 1) == (0x1u64 & (b >> 13) == 1),
            (0x1u64 & (a >> 14) == 1) == (0x1u64 & (b >> 14) == 1),
            (0x1u64 & (a >> 15) == 1) == (0x1u64 & (b >> 15) == 1),
            (0x1u64 & (a >> 16) == 1) == (0x1u64 & (b >> 16) == 1),
            (0x1u64 & (a >> 17) == 1) == (0x1u64 & (b >> 17) == 1),
            (0x1u64 & (a >> 18) == 1) == (0x1u64 & (b >> 18) == 1),
            (0x1u64 & (a >> 19) == 1) == (0x1u64 & (b >> 19) == 1),
            (0x1u64 & (a >> 20) == 1) == (0x1u64 & (b >> 20) == 1),
            (0x1u64 & (a >> 21) == 1) == (0x1u64 & (b >> 21) == 1),
            (0x1u64 & (a >> 22) == 1) == (0x1u64 & (b >> 22) == 1),
            (0x1u64 & (a >> 23) == 1) == (0x1u64 & (b >> 23) == 1),
            (0x1u64 & (a >> 24) == 1) == (0x1u64 & (b >> 24) == 1),
            (0x1u64 & (a >> 25) == 1) == (0x1u64 & (b >> 25) == 1),
            (0x1u64 & (a >> 26) == 1) == (0x1u64 & (b >> 26) == 1),
            (0x1u64 & (a >> 27) == 1) == (0x1u64 & (b >> 27) == 1),
            (0x1u64 & (a >> 28) == 1) == (0x1u64 & (b >> 28) == 1),
            (0x1u64 & (a >> 29) == 1) == (0x1u64 & (b >> 29) == 1),
            (0x1u64 & (a >> 30) == 1) == (0x1u64 & (b >> 30) == 1),
            (0x1u64 & (a >> 31) == 1) == (0x1u64 & (b >> 31) == 1),
            (0x1u64 & (a >> 32) == 1) == (0x1u64 & (b >> 32) == 1),
            (0x1u64 & (a >> 33) == 1) == (0x1u64 & (b >> 33) == 1),
            (0x1u64 & (a >> 34) == 1) == (0x1u64 & (b >> 34) == 1),
            (0x1u64 & (a >> 35) == 1) == (0x1u64 & (b >> 35) == 1),
            (0x1u64 & (a >> 36) == 1) == (0x1u64 & (b >> 36) == 1),
            (0x1u64 & (a >> 37) == 1) == (0x1u64 & (b >> 37) == 1),
            (0x1u64 & (a >> 38) == 1) == (0x1u64 & (b >> 38) == 1),
            (0x1u64 & (a >> 39) == 1) == (0x1u64 & (b >> 39) == 1),
            (0x1u64 & (a >> 40) == 1) == (0x1u64 & (b >> 40) == 1),
            (0x1u64 & (a >> 41) == 1) == (0x1u64 & (b >> 41) == 1),
            (0x1u64 & (a >> 42) == 1) == (0x1u64 & (b >> 42) == 1),
            (0x1u64 & (a >> 43) == 1) == (0x1u64 & (b >> 43) == 1),
            (0x1u64 & (a >> 44) == 1) == (0x1u64 & (b >> 44) == 1),
            (0x1u64 & (a >> 45) == 1) == (0x1u64 & (b >> 45) == 1),
            (0x1u64 & (a >> 46) == 1) == (0x1u64 & (b >> 46) == 1),
            (0x1u64 & (a >> 47) == 1) == (0x1u64 & (b >> 47) == 1),
            (0x1u64 & (a >> 48) == 1) == (0x1u64 & (b >> 48) == 1),
            (0x1u64 & (a >> 49) == 1) == (0x1u64 & (b >> 49) == 1),
            (0x1u64 & (a >> 50) == 1) == (0x1u64 & (b >> 50) == 1),
            (0x1u64 & (a >> 51) == 1) == (0x1u64 & (b >> 51) == 1),
            (0x1u64 & (a >> 52) == 1) == (0x1u64 & (b >> 52) == 1),
            (0x1u64 & (a >> 53) == 1) == (0x1u64 & (b >> 53) == 1),
            (0x1u64 & (a >> 54) == 1) == (0x1u64 & (b >> 54) == 1),
            (0x1u64 & (a >> 55) == 1) == (0x1u64 & (b >> 55) == 1),
            (0x1u64 & (a >> 56) == 1) == (0x1u64 & (b >> 56) == 1),
            (0x1u64 & (a >> 57) == 1) == (0x1u64 & (b >> 57) == 1),
            (0x1u64 & (a >> 58) == 1) == (0x1u64 & (b >> 58) == 1),
            (0x1u64 & (a >> 59) == 1) == (0x1u64 & (b >> 59) == 1),
            (0x1u64 & (a >> 60) == 1) == (0x1u64 & (b >> 60) == 1),
            (0x1u64 & (a >> 61) == 1) == (0x1u64 & (b >> 61) == 1),
            (0x1u64 & (a >> 62) == 1) == (0x1u64 & (b >> 62) == 1),
            (0x1u64 & (a >> 63) == 1) == (0x1u64 & (b >> 63) == 1);
}

pub proof fn casting128_to_64(val: u128, i: u128)
    requires i < 64
    ensures get_bit_u128(val,i) == get_bit_u64(val as u64,i as u64)
{
    let b = get_bit_u128(val,i);
    let val2 = val & 0xffffffffffffffff;
    let b2 = get_bit_u128(val2,i);
    let valp = val as u64;
    let bp = get_bit_u64(valp,i as u64);

    assert(valp == val2) by (bit_vector)
        requires
            valp == val as u64,
            val2 == val & 0xffffffffffffffff;
    assert(b == b2) by (bit_vector)
        requires 
            val2 == val & 0xffffffffffffffff,
            b == (0x1u128 & (val >> i) == 1),
            b2 == (0x1u128 & (val2 >> i) == 1),
            i < 64;
}

}
