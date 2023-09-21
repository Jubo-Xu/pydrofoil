import pytest
import sys

from pydrofoil import supportcode
from pydrofoil import bitvector
from pydrofoil.bitvector import Integer, SmallInteger, BigInteger, SmallBitVector, GenericBitVector, SparseBitVector
from hypothesis import given, strategies, assume, example, settings
from rpython.rlib.rarithmetic import r_uint, intmask, r_ulonglong
from rpython.rlib.rbigint import rbigint

MININT = -sys.maxint - 1

def make_int(data):
    if data.draw(strategies.booleans()):
        # big ints
        return BigInteger(rbigint.fromlong(data.draw(strategies.integers())))
    else:
        # small ints
        return SmallInteger(data.draw(ints))

ints = strategies.integers(-sys.maxint-1, sys.maxint)
wrapped_ints = strategies.builds(
        make_int,
        strategies.data())

def gbv(size, val):
    return bitvector.GenericBitVector(size, rbigint.fromlong(val))

def bv(size, val):
    return bitvector.from_ruint(size, r_uint(val))

def si(val):
    return bitvector.SmallInteger(val)

def bi(val):
    return bitvector.BigInteger(rbigint.fromlong(val))

machine = "dummy"

def test_fast_signed():
    assert supportcode.fast_signed(machine, 0b0, 1) == 0
    assert supportcode.fast_signed(machine, 0b1, 1) == -1
    assert supportcode.fast_signed(machine, 0b0, 2) == 0
    assert supportcode.fast_signed(machine, 0b1, 2) == 1
    assert supportcode.fast_signed(machine, 0b10, 2) == -2
    assert supportcode.fast_signed(machine, 0b11, 2) == -1

def test_signed():
    for c in gbv, bv:
        assert supportcode.sail_signed(machine, c(1, 0b0)).toint() == 0
        assert supportcode.sail_signed(machine, c(1, 0b1)).toint() == -1
        assert supportcode.sail_signed(machine, c(2, 0b0)).toint() == 0
        assert supportcode.sail_signed(machine, c(2, 0b1)).toint() == 1
        assert supportcode.sail_signed(machine, c(2, 0b10)).toint() == -2
        assert supportcode.sail_signed(machine, c(2, 0b11)).toint() == -1
        assert supportcode.sail_signed(machine, c(64, 0xffffffffffffffff)).toint() == -1
        assert supportcode.sail_signed(machine, c(64, 0x1)).toint() == 1

def test_sign_extend():
    for c in gbv, bv:
        assert supportcode.sign_extend(machine, c(1, 0b0), Integer.fromint(2)).toint() == 0
        assert supportcode.sign_extend(machine, c(1, 0b1), Integer.fromint(2)).toint() == 0b11
        assert supportcode.sign_extend(machine, c(2, 0b00), Integer.fromint(4)).toint() == 0
        assert supportcode.sign_extend(machine, c(2, 0b01), Integer.fromint(4)).toint() == 1
        assert supportcode.sign_extend(machine, c(2, 0b10), Integer.fromint(4)).toint() == 0b1110
        assert supportcode.sign_extend(machine, c(2, 0b11), Integer.fromint(4)).toint() == 0b1111

        assert supportcode.sign_extend(machine, c(2, 0b00), Integer.fromint(100)).tobigint().tolong() == 0
        assert supportcode.sign_extend(machine, c(2, 0b01), Integer.fromint(100)).tobigint().tolong() == 1
        assert supportcode.sign_extend(machine, c(2, 0b10), Integer.fromint(100)).tobigint().tolong() == 0b1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111110
        assert supportcode.sign_extend(machine, c(2, 0b11), Integer.fromint(100)).tobigint().tolong() == 0b1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111

# def test_zero_extend():
#     for c in gbv, bv:
#         assert supportcode.zero_extend(machine, c(1, 0b0), Integer.fromint(2)).size() == 2
#         assert supportcode.zero_extend(machine, c(2, 0b00), Integer.fromint(100)).size() == 100
#         assert supportcode.zero_extend(machine, c(1, 0b0), Integer.fromint(2)).toint() == 0
#         assert supportcode.zero_extend(machine, c(1, 0b1), Integer.fromint(2)).toint() == 0b01
#         assert supportcode.zero_extend(machine, c(2, 0b00), Integer.fromint(4)).toint() == 0
#         assert supportcode.zero_extend(machine, c(2, 0b01), Integer.fromint(4)).toint() == 1
#         assert supportcode.zero_extend(machine, c(2, 0b10), Integer.fromint(4)).toint() == 0b0010
#         assert supportcode.zero_extend(machine, c(2, 0b11), Integer.fromint(4)).toint() == 0b0011

#         assert supportcode.zero_extend(machine, c(2, 0b00), Integer.fromint(100)).tobigint().tolong() == 0
#         assert supportcode.zero_extend(machine, c(2, 0b01), Integer.fromint(100)).tobigint().tolong() == 1
#         assert supportcode.zero_extend(machine, c(2, 0b10), Integer.fromint(100)).tobigint().tolong() == 0b0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010
#         assert supportcode.zero_extend(machine, c(2, 0b11), Integer.fromint(100)).tobigint().tolong() == 0b0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011

def test_unsigned():
    for c in gbv, bv:
        x = c(8, 0b10001101)
        assert x.unsigned().tolong() == 0b10001101
        x = c(64, 0b10001101)
        assert x.unsigned().tolong() == 0b10001101
        x = c(64, r_uint(-1))
        assert x.unsigned().tolong() == (1<<64)-1

def test_get_slice_int():
    for c in si, bi:
        assert supportcode.get_slice_int(machine, Integer.fromint(8), c(0b011010010000), Integer.fromint(4)).tolong() == 0b01101001
        assert supportcode.get_slice_int(machine, Integer.fromint(8), c(-1), Integer.fromint(4)).tolong() == 0b11111111
        assert supportcode.get_slice_int(machine, Integer.fromint(64), c(-1), Integer.fromint(5)).tolong() == 0xffffffffffffffff
        assert supportcode.get_slice_int(machine, Integer.fromint(100), c(-1), Integer.fromint(11)).tolong() == 0b1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
        assert supportcode.get_slice_int(machine, Integer.fromint(8), c(-1), Integer.fromint(1000)).tolong() == 0b11111111
        assert supportcode.get_slice_int(machine, Integer.fromint(64), c(-1), Integer.fromint(1000)).tolong() == 0xffffffffffffffff
        assert supportcode.get_slice_int(machine, Integer.fromint(100), c(-1), Integer.fromint(1000)).tolong() == 0b1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111


def test_vector_access():
    for c in gbv, bv:
        x = c(6, 0b101100)
        for i in range(6):
            assert x.read_bit(i) == [0, 0, 1, 1, 0, 1][i]

def test_vector_update():
    for c in gbv, bv:
        x = c(6, 1)
        res = x.update_bit(2, 1)
        assert res.size() == 6
        assert res.toint() == 0b101

        x = c(6, 1)
        res = x.update_bit(0, 1)
        assert res.size() == 6
        assert res.toint() == 0b1

        x = c(6, 0b11)
        res = x.update_bit(2, 0)
        assert res.size() == 6
        assert res.toint() == 0b011

        x = c(6, 0b111)
        res = x.update_bit(1, 0)
        assert res.size() == 6
        assert res.toint() == 0b101

def test_vector_subrange():
    for c in gbv, bv:
        x = c(6, 0b111)
        r = x.subrange(3, 2)
        assert r.size() == 2
        assert r.toint() == 1
        assert isinstance(r, bitvector.SmallBitVector)

    # regression bug
    b = gbv(128, 0x36000000000000001200L)
    x = b.subrange(63, 0)
    assert x.touint() == 0x1200

    b = gbv(128, 0x36000000000000001200L)
    x = b.subrange(66, 0)
    assert x.tolong() == 0x1200
    assert isinstance(x, bitvector.GenericBitVector)

@given(strategies.data())
def test_hypothesis_vector_subrange(data):
    bitwidth = data.draw(strategies.integers(1, 10000))
    lower = data.draw(strategies.integers(0, bitwidth-1))
    upper = data.draw(strategies.integers(lower, bitwidth-1))
    value = data.draw(strategies.integers(0, 2**bitwidth - 1))
    as_bit_string = bin(value)[2:]
    assert len(as_bit_string) <= bitwidth
    as_bit_string = as_bit_string.rjust(bitwidth, '0')[::-1]
    correct_res = as_bit_string[lower:upper+1] # sail is inclusive
    correct_res_as_int = int(correct_res[::-1], 2)

    # now do the sail computation
    bv = bitvector.from_bigint(bitwidth, rbigint.fromlong(value))
    bvres = bv.subrange(upper, lower)
    assert bvres.tobigint().tolong() == correct_res_as_int

@settings(deadline=1000)
@given(strategies.data())
def test_hypothesis_sign_extend(data):
    bitwidth = data.draw(strategies.integers(1, 10000))
    target_bitwidth = bitwidth + data.draw(strategies.integers(1, 100))
    value = data.draw(strategies.integers(0, 2**bitwidth - 1))
    bv = bitvector.from_bigint(bitwidth, rbigint.fromlong(value))
    res = bv.sign_extend(target_bitwidth)
    print bitwidth, target_bitwidth, value, bv, res, bv.signed().tobigint(), res.signed().tobigint()
    assert bv.signed().tobigint().tolong() == res.signed().tobigint().tolong()

@given(strategies.data())
def test_hypothesis_vector_subrange_unwrapped_res(data):
    if data.draw(strategies.booleans()):
        bitwidth = data.draw(strategies.integers(65, 10000))
    else:
        bitwidth = data.draw(strategies.integers(1, 64))
    lower = data.draw(strategies.integers(0, bitwidth-1))
    upper = data.draw(strategies.integers(lower, min(bitwidth-1, lower + 63)))
    assert 1 <= upper - lower + 1 <= 64
    assert 0 <= lower <= upper < bitwidth
    value = data.draw(strategies.integers(0, 2**bitwidth - 1))
    as_bit_string = bin(value)[2:]
    assert len(as_bit_string) <= bitwidth
    as_bit_string = as_bit_string.rjust(bitwidth, '0')[::-1]
    correct_res = as_bit_string[lower:upper+1] # sail is inclusive
    correct_res_as_int = int(correct_res[::-1], 2)

    # now do the sail computation
    bv = bitvector.from_bigint(bitwidth, rbigint.fromlong(value))
    bvres = bv.subrange_unwrapped_res(upper, lower)
    assert bvres == correct_res_as_int

@given(strategies.data())
def test_hypothesis_rbigint_extract_ruint(data):
    bitwidth = data.draw(strategies.integers(1, 10000))
    start = data.draw(strategies.integers(0, 2 * bitwidth))
    value = data.draw(strategies.integers(0, 2**bitwidth - 1))
    rb = rbigint.fromlong(value)
    bv = bitvector.from_bigint(bitvector, rb)
    res = bv.subrange_unwrapped_res(start + 63, start)
    assert res == rb.rshift(start).and_(rbigint.fromlong(2**64-1)).tolong()


def test_vector_update_subrange():
    for c1 in gbv, bv:
        for c2 in gbv, bv:
            x = c1(8, 0b10001101)
            x = x.update_subrange(5, 2, c2(4, 0b1010))
            assert x.toint() == 0b10101001
            x = c1(64, 0b10001101)
            y = c2(64, 0b1101001010010)
            x = x.update_subrange(63, 0, y)
            assert x.eq(y)

def test_sparse_vector_update_subrange():
    for c in SparseBitVector, gbv:
        x = c(100, 0b10001101)
        x = x.update_subrange(5, 2, bv(4, 0b1010))
        assert x.toint() == 0b10101001
        x = c(100, 0b10001101)
        y = c(100, 0b1101001010010)
        x = x.update_subrange(99, 0, y)
        assert x.eq(y)
        x = SparseBitVector(65, 0b10001101)
        y = c(65, 0b1101001010010)
        x = x.update_subrange(64, 0, y)
        assert x.eq(y)
    x = SparseBitVector(1000, 0b10001101)
    y = gbv(1000, 0b1101001010010)
    x = x.update_subrange(999, 0, y)
    assert isinstance(x, bitvector.GenericBitVector)

@given(strategies.data())
def test_sparse_vector_update_subrange_hypothesis(data):
    width = data.draw(strategies.integers(65, 256))
    value = r_uint(data.draw(strategies.integers(0, 2**64 - 1)))
    lower = data.draw(strategies.integers(0, width-1))
    upper = data.draw(strategies.integers(lower, width-1))
    subwidth = upper - lower + 1
    subvalue = r_uint(data.draw(strategies.integers(0, 2 ** min(subwidth, 64) - 1)))
    replace_bv = bitvector.from_ruint(subwidth, subvalue)

    # two checks: check that the generic is the same as sparse
    sbv = SparseBitVector(width, value)
    sres = sbv.update_subrange(upper, lower, replace_bv)
    sres2 = sbv._to_generic().update_subrange(upper, lower, replace_bv)
    assert sres.eq(sres2)
    # second check: the read of the same range must return replace_bv
    assert replace_bv.eq(sres.subrange(upper, lower))

def test_vector_shift():
    for c in gbv, bv:
        x = c(8, 0b10001101)
        res = x.lshift(5)
        assert res.size() == 8
        assert res.toint() == 0b10100000

        x = c(8, 0b10001101)
        res = x.rshift(5)
        assert res.size() == 8
        assert res.toint() == 0b00000100

        x = c(8, 0b10001101)
        res = x.lshift(65)
        assert res.size() == 8
        assert res.toint() == 0

        x = c(8, 0b10001101)
        res = x.rshift(65)
        assert res.size() == 8
        assert res.toint() == 0

def test_vector_shift_bits():
    for c in gbv, bv:
        x = c(8, 0b10001101)
        res = x.lshift_bits(c(8, 5))
        assert res.size() == 8
        assert res.toint() == 0b10100000

        x = c(8, 0b10001101)
        res = x.rshift_bits(c(16, 5))
        assert res.size() == 8
        assert res.toint() == 0b00000100

        x = c(8, 0b10001101)
        res = x.lshift_bits(c(8, 65))
        assert res.size() == 8
        assert res.toint() == 0

        x = c(8, 0b10001101)
        res = x.rshift_bits(c(16, 65))
        assert res.size() == 8
        assert res.toint() == 0
def test_arith_shiftr():
    for c in bv, gbv:
        x = c(8, 0b10001101)
        res = x.arith_rshift(2)
        assert res.size() == 8
        assert res.toint() == 0b11100011

        res = x.arith_rshift(8)
        assert res.size() == 8
        assert res.toint() == 0b11111111

        x = c(8, 0b00101101)
        res = x.arith_rshift(3)
        assert res.size() == 8
        assert res.toint() == 0b101

@given(strategies.data())
def test_arith_shiftr_hypothesis(data):
    small = data.draw(strategies.booleans())
    if small:
        size = data.draw(strategies.integers(1, 64))
        value = data.draw(strategies.integers(0, 2**size-1))
        bv = bitvector.SmallBitVector(size, r_uint(value))
    else:
        size = data.draw(strategies.integers(65, 5000))
        value = data.draw(strategies.integers(0, 2**size-1))
        bv = bitvector.GenericBitVector(size, rbigint.fromlong(value))
    shift = data.draw(strategies.integers(0, size+10))
    res = bv.arith_rshift(shift)
    # compare against signed, then integer shift
    intres = bv.signed().tobigint().tolong() >> shift
    assert res.tobigint().tolong() == intres & ((1 << size) - 1)

def test_bitvector_touint():
    for size in [6, 6000]:
        assert bv(size, 0b11).touint() == r_uint(0b11)

def test_add_int():
    for c in bi, si:
        assert bv(6, 0b11).add_int(c(0b111111111)).touint() == (0b11 + 0b111111111) & 0b111111
        assert gbv(6000, 0b11).add_int(c(0b111111111)).touint() == 0b11 + 0b111111111

def test_add_bits_int_bv_i():
    assert supportcode.add_bits_int_bv_i(None, r_uint(0b11), 6, 0b111111111) == (0b11 + 0b111111111) & 0b111111
    assert supportcode.add_bits_int_bv_i(None, r_uint(0b11), 6, -0b111111111) == (0b11 - 0b111111111) & 0b111111
    assert supportcode.add_bits_int_bv_i(None, r_uint(0b1011), 6, -2 ** 63) == (0b1011 - 2**63) & 0b111111

def test_bv_bitwise():
    for c in gbv, bv:
        i1 = c(8, 0b11110000)
        i2 = c(8, 0b11001100)
        res = i1.and_(i2)
        assert res.toint() == 0b11110000 & 0b11001100
        res = i1.or_(i2)
        assert res.toint() == 0b11110000 | 0b11001100
        res = i1.xor(i2)
        assert res.toint() == 0b11110000 ^ 0b11001100
        res = i1.invert()
        assert res.toint() == 0b00001111

def test_add_bv():
    for c in gbv, bv:
        assert supportcode.add_bits(None, c(6, 0b11), c(6, 0b111)).touint() == (0b11 + 0b111) & 0b111111
        assert supportcode.add_bits(None, c(6, 0b10000), c(6, 0b10001)).touint() == (0b10000 + 0b10001) & 0b111111
        assert supportcode.add_bits(None, c(6, 0b100000), c(6, 0b100001)).touint() == (0b100000 + 0b100001) & 0b111111

def test_sub_bv():
    for c in gbv, bv:
        assert supportcode.sub_bits(None, c(6, 0b111), c(6, 0b11)).touint() == (0b111 - 0b11) & 0b111111
        assert supportcode.sub_bits(None, c(6, 0b10000), c(6, 0b10001)).touint() == (0b10000 - 0b10001) & 0b111111
        assert supportcode.sub_bits(None, c(6, 0b100000), c(6, 0b100001)).touint() == (0b100000 - 0b100001) & 0b111111


def test_eq_int():
    for c1 in bi, si:
        for c2 in bi, si:
            assert c1(-12331).eq(c2(-12331))
            assert not c1(-12331).eq(c2(12331))

def test_op_int():
    for c1 in bi, si:
        for c2 in bi, si:
            for v1 in [-10, 223, 12311, 0, 1, 2**63-1]:
                a = c1(v1)
                for v2 in [-10, 223, 12311, 0, 1, 2**63-1, -2**45]:
                    b = c2(v2)
                    assert a.add(b).tolong() == v1 + v2
                    assert a.sub(b).tolong() == v1 - v2
                    assert a.mul(b).tolong() == v1 * v2
                    if v2:
                        assert c1(abs(v1)).tdiv(c2(abs(v2))).tolong() == abs(v1) // abs(v2)
                        assert c1(abs(v1)).tmod(c2(abs(v2))).tolong() == abs(v1) % abs(v2)
                        # (a/b) * b + a%b == a
                        assert a.tdiv(b).mul(b).add(a.tmod(b)).eq(a)

                    assert a.eq(b) == (v1 == v2)
                    assert a.lt(b) == (v1 < v2)
                    assert a.gt(b) == (v1 > v2)
                    assert a.le(b) == (v1 <= v2)
                    assert a.ge(b) == (v1 >= v2)
                with pytest.raises(ZeroDivisionError):
                    c1(v1).tdiv(c2(0))
                    c1(v1).tmod(c2(0))

@given(wrapped_ints, wrapped_ints)
def test_op_int_hypothesis(a, b):
    v1 = a.tobigint().tolong()
    v2 = b.tobigint().tolong()
    assert a.add(b).tolong() == v1 + v2
    assert a.sub(b).tolong() == v1 - v2
    assert a.mul(b).tolong() == v1 * v2
    if v2:
        assert a.abs().tdiv(b.abs()).tolong() == abs(v1) // abs(v2)
        assert a.abs().tmod(b.abs()).tolong() == abs(v1) % abs(v2)
        # (a/b) * b + a%b == a
        assert a.tdiv(b).mul(b).add(a.tmod(b)).eq(a)

    assert a.eq(b) == (v1 == v2)
    assert a.lt(b) == (v1 < v2)
    assert a.gt(b) == (v1 > v2)
    assert a.le(b) == (v1 <= v2)
    assert a.ge(b) == (v1 >= v2)
    with pytest.raises(ZeroDivisionError):
        a.tdiv(si(0))
    with pytest.raises(ZeroDivisionError):
        a.tdiv(si(0))
    with pytest.raises(ZeroDivisionError):
        a.tmod(bi(0))
    with pytest.raises(ZeroDivisionError):
        a.tmod(bi(0))

def test_op_int_div_mod():
    for c1 in bi, si:
        for c2 in bi, si:
            # round towards zero
            assert c1(1).tdiv(c2(2)).tolong() == 0
            assert c1(-1).tdiv(c2(2)).tolong() == 0
            assert c1(1).tdiv(c2(-2)).tolong() == 0
            assert c1(-1).tdiv(c2(-2)).tolong() == 0

            # mod signs
            assert c1(5).tmod(c2(3)).tolong() == 2
            assert c1(5).tmod(c2(-3)).tolong() == 2
            assert c1(-5).tmod(c2(3)).tolong() == -2
            assert c1(-5).tmod(c2(-3)).tolong() == -2

            # ovf correctly
            assert c1(-2**63).tdiv(c2(-1)).tolong() == 2 ** 63
            assert c1(-2**63).tmod(c2(-1)).tolong() == 0

def test_shift_amount():
    for i in range(63):
        assert BigInteger._shift_amount(2 ** i) == i

def test_mul_optimized(monkeypatch):
    monkeypatch.setattr(rbigint, "mul", None)
    monkeypatch.setattr(rbigint, "int_mul", None)
    res = bi(3 ** 100).mul(si(16))
    assert res.tolong() == 3 ** 100 * 16
    res = si(1024).mul(bi(-5 ** 60))
    assert res.tolong() == -5 ** 60 * 1024

def test_op_gv_int():
    for c1 in gbv, bv:
        for c2 in bi, si:
            assert c1(16, 4).add_int(c2(9)).touint() == 13
            assert c1(16, 4).sub_int(c2(9)).touint() == r_uint((-5) & 0xffff)

def test_int_shift():
    for c in bi, si:
        assert c(0b1010001).rshift(2).tobigint().tolong() == 0b10100
        assert c(-0b1010001).rshift(3).tobigint().tolong() == -0b1011
        assert c(0b1010001).lshift(2).tobigint().tolong() == 0b101000100
        assert c(-0b1010001).lshift(3).tobigint().tolong() == -0b1010001000

def test_replicate_bits():
    for c1 in gbv, bv:
        res = c1(3, 0b011).replicate(10)
        assert res.size() == 3 * 10
        assert res.touint() == 0b011011011011011011011011011011
        res = c1(8, 0xe7).replicate(15)
        assert res.size() == 8*15
        assert res.tobigint().tolong() == 0xe7e7e7e7e7e7e7e7e7e7e7e7e7e7e7

def test_truncate():
    for c1 in gbv, bv:
        res = c1(10, 0b1011010100).truncate(2)
        assert res.size() == 2
        assert res.touint() == 0b00
        res = c1(10, 0b1011010100).truncate(6)
        assert res.size() == 6
        assert res.touint() == 0b010100

@given(strategies.data())
def test_hypothesis_truncate(data):
    if not data.draw(strategies.booleans()):
        bitwidth = data.draw(strategies.integers(1, 64))
        truncatewidth = data.draw(strategies.integers(1, bitwidth))
    else:
        bitwidth = data.draw(strategies.integers(65, 10000))
        if not data.draw(strategies.booleans()):
            truncatewidth = data.draw(strategies.integers(1, 64))
        else:
            truncatewidth = data.draw(strategies.integers(1, bitwidth))
    value = data.draw(strategies.integers(0, 2**bitwidth - 1))
    as_bit_string = bin(value)[2:]
    bv = bitvector.from_bigint(bitwidth, rbigint.fromlong(value))
    res = bv.truncate(truncatewidth)
    assert bin(bv.tolong())[2:].rjust(bitwidth, '0')[-truncatewidth:] == bin(res.tolong())[2:].rjust(truncatewidth, '0')


def test_string_of_bits():
    for c in gbv, bv:
        assert c(32, 0x1245ab).string_of_bits() == "0x001245AB"
        assert c(64, 0x1245ab).string_of_bits() == "0x00000000001245AB"
        assert c(3, 0b1).string_of_bits() == "0b001"
        assert c(9, 0b1101).string_of_bits() == "0b000001101"

def test_append():
    for c1 in gbv, bv:
        for c2 in gbv, bv:
            assert c1(16, 0xa9e3).append(c2(16, 0x04fb)).toint() == 0xa9e304fb

def test_abs_int():
    for c in si, bi:
        for value in [-2**63, -6, 10, 2**63-1]:
            assert c(value).abs().tobigint().tolong() == abs(value)

def test_rshift_int():
   for c in bi, si:
       assert c(0b1010001).rshift(2).tobigint().tolong() == 0b10100
       assert c(-0b1010001).rshift(3).tobigint().tolong() == -11

def test_emod_ediv_int():
   for c1 in bi, si:
        for c2 in bi, si:
            assert c1(123875).emod(si(13)).toint() == 123875 % 13
            assert c1(123875).ediv(c2(13)).toint() == 123875 // 13
            assert c1(MININT).ediv(c2(2)).toint() == -2**62
            assert c1(MININT).ediv(c2(-2)).toint() == 2**62
            assert c1(MININT).ediv(c2(MININT)).toint() == 1
            assert c1(5).ediv(c2(MININT)).toint() == 0
            assert c1(-5).ediv(c2(MININT)).toint() == 1
            assert c1(MININT + 1).ediv(c2(sys.maxint)).toint() == -1
            assert c1(MININT).ediv(c2(MININT)).toint() == 1
            assert c1(7).ediv(c2(5)).toint() == 1
            assert c1(7).ediv(c2(-5)).toint() == -1
            assert c1(-7).ediv(c2(-5)).toint() == 2
            assert c1(-7).ediv(c2(5)).toint() == -2
            assert c1(12).ediv(c2(3)).toint() == 4
            assert c1(12).ediv(c2(-3)).toint() == -4
            assert c1(-12).ediv(c2(3)).toint() == -4
            assert c1(-12).ediv(c2(-3)).toint() == 4
            assert c1(MININT).emod(c2(2)).toint() == 0
            assert c1(MININT).emod(c2(- 2)).toint() == 0
            assert c1(MININT).emod(c2(- 2 ** 63)).toint() == 0
            assert c1(sys.maxint).emod(c2(sys.maxint)).toint() == 0
            assert c1(7).emod(c2(5)).toint() == 2
            assert c1(7).emod(c2(-5)).toint() == 2
            assert c1(-7).emod(c2(5)).toint() == 3
            assert c1(-7).emod(c2(-5)).toint() == 3
            assert c1(12).emod(c2(3)).toint() == 0
            assert c1(12).emod(c2(-3)).toint() == 0
            assert c1(-12).emod(c2(3)).toint() == 0
            assert c1(-12).emod(c2(-3)).toint() == 0


   assert bi(0xfffffe00411e0e90L).emod(si(64)).toint() == 16
   assert bi(98765432109876543210).ediv(bi(12345678901234567890)).toint() == 8
   assert bi(98765432109876543210).emod(bi(12345678901234567890)).toint() == 900000000090
   assert bi(12345678901234567890).ediv(bi(-10000000000000000000)).toint() == -1
   assert bi(12345678901234567890).emod(bi(-10000000000000000000)).toint() == 2345678901234567890
   assert bi(-12345678901234567890).ediv(bi(-10000000000000000000)).toint() == 2
   assert bi(-12345678901234567890).ediv(bi(10000000000000000000)).toint() == -2
   assert bi(-12345678901234567890).emod(bi(10000000000000000000)).toint() == 7654321098765432110
   assert bi(-12345678901234567890).emod(bi(-10000000000000000000)).toint() == 7654321098765432110

def test_pow2():
    for i in range(1000):
        assert supportcode.pow2_i(None, i).tobigint().tolong() == 2 ** i
    # check that small results use small ints
    for i in range(63):
        assert supportcode.pow2_i(None, i).val == 2 ** i

# softfloat

class DummyMachine(object): pass


def test_softfloat_f16add():
    machine = DummyMachine()
    supportcode.softfloat_f16add(machine, 0, 0b0011110000000000, 0b0011100000000000)
    assert machine._reg_zfloat_result == 0b0011111000000000

def test_softfloat_f16div():
    machine = DummyMachine()
    supportcode.softfloat_f16div(machine, 0, 0b0011110000000000, 0b0011100000000000)
    assert machine._reg_zfloat_result == 0b0100000000000000

def test_softfloat_f16eq():
    machine = DummyMachine()
    supportcode.softfloat_f16eq(machine, 0b0011100000000000, 0b0011100000000000)
    assert machine._reg_zfloat_result == 1

def test_softfloat_f16le():
    machine = DummyMachine()
    supportcode.softfloat_f16le(machine, 0b0011100000000000, 0b0011100000000000)
    assert machine._reg_zfloat_result == 1

def test_softfloat_f16lt():
    machine = DummyMachine()
    supportcode.softfloat_f16lt(machine, 0b0011100000000000, 0b0011100000000000)
    assert machine._reg_zfloat_result == 0

def test_softfloat_f16mul():
    machine = DummyMachine()
    supportcode.softfloat_f16mul(machine, 0, 0b0011110000000000, 0b0011100000000000)
    assert machine._reg_zfloat_result == 0b0011100000000000

def test_softfloat_f16muladd():
    machine = DummyMachine()
    supportcode.softfloat_f16muladd(machine, 0, 0b0011110000000000, 0b0011100000000000, 0b0011110000000000)
    assert machine._reg_zfloat_result == 0b0011111000000000

def test_softfloat_f16sqrt():
    machine = DummyMachine()
    supportcode.softfloat_f16sqrt(machine, 0, 0b0100010000000000)
    assert machine._reg_zfloat_result == 0b0100000000000000

def test_softfloat_f16sub():
    machine = DummyMachine()
    supportcode.softfloat_f16sub(machine, 0, 0b0011110000000000, 0b0011100000000000)
    assert machine._reg_zfloat_result == 0b0011100000000000

def test_softfloat_f32add():
    machine = DummyMachine()
    supportcode.softfloat_f32add(machine, 0, 0b00111111000000000000000000000000, 0b00111111100000000000000000000000)
    assert machine._reg_zfloat_result == 0b00111111110000000000000000000000

def test_softfloat_f32div():
    machine = DummyMachine()
    supportcode.softfloat_f32div(machine, 0, 0b00111111100000000000000000000000, 0b01000000000000000000000000000000)
    assert machine._reg_zfloat_result == 0b00111111000000000000000000000000

def test_softfloat_f32eq():
    machine = DummyMachine()
    supportcode.softfloat_f32eq(machine, 0b00111111000000000000000000000000, 0b00111111000000000000000000000000)
    assert machine._reg_zfloat_result == 1

def test_softfloat_f32le():
    machine = DummyMachine()
    supportcode.softfloat_f32le(machine, 0b00111111000000000000000000000000, 0b00111111000000000000000000000000)
    assert machine._reg_zfloat_result == 1

def test_softfloat_f32lt():
    machine = DummyMachine()
    supportcode.softfloat_f32lt(machine, 0b00111111000000000000000000000000, 0b00111111000000000000000000000000)
    assert machine._reg_zfloat_result == 0

def test_softfloat_f32mul():
    machine = DummyMachine()
    supportcode.softfloat_f32mul(machine, 0, 0b00111111100000000000000000000000, 0b01000000000000000000000000000000)
    assert machine._reg_zfloat_result == 0b01000000000000000000000000000000

def test_softfloat_f32muladd():
    machine = DummyMachine()
    supportcode.softfloat_f32muladd(machine, 0, 0b00111111100000000000000000000000, 0b00111111000000000000000000000000, 0b00111111100000000000000000000000)
    assert machine._reg_zfloat_result == 0b00111111110000000000000000000000

def test_softfloat_f32sqrt():
    machine = DummyMachine()
    supportcode.softfloat_f32sqrt(machine, 0, 0b01000000100000000000000000000000)
    assert machine._reg_zfloat_result == 0b01000000000000000000000000000000

def test_softfloat_f32sub():
    machine = DummyMachine()
    supportcode.softfloat_f32sub(machine, 0, 0b01000000000000000000000000000000, 0b00111111100000000000000000000000)
    assert machine._reg_zfloat_result == 0b00111111100000000000000000000000

def test_softfloat_f64add():
    machine = DummyMachine()
    supportcode.softfloat_f64add(machine, 0, 0b0011111111110000000000000000000000000000000000000000000000000000, 0b0011111111100000000000000000000000000000000000000000000000000000)
    assert machine._reg_zfloat_result == 0b0011111111111000000000000000000000000000000000000000000000000000

def test_softfloat_f64div():
    machine = DummyMachine()
    supportcode.softfloat_f64div(machine, 0, 0b0011111111110000000000000000000000000000000000000000000000000000, 0b0011111111100000000000000000000000000000000000000000000000000000)
    assert machine._reg_zfloat_result == 0b0100000000000000000000000000000000000000000000000000000000000000

def test_softfloat_f64eq():
    machine = DummyMachine()
    supportcode.softfloat_f64eq(machine, 0b0011111111100000000000000000000000000000000000000000000000000000, 0b0011111111100000000000000000000000000000000000000000000000000000)
    assert machine._reg_zfloat_result == 1

def test_softfloat_f64le():
    machine = DummyMachine()
    supportcode.softfloat_f64le(machine, 0b0011111111100000000000000000000000000000000000000000000000000000, 0b0011111111100000000000000000000000000000000000000000000000000000)
    assert machine._reg_zfloat_result == 1

def test_softfloat_f64lt():
    machine = DummyMachine()
    supportcode.softfloat_f64lt(machine, 0b0011111111100000000000000000000000000000000000000000000000000000, 0b0011111111100000000000000000000000000000000000000000000000000000)
    assert machine._reg_zfloat_result == 0

def test_softfloat_f64mul():
    machine = DummyMachine()
    supportcode.softfloat_f64mul(machine, 0, 0b0011111111110000000000000000000000000000000000000000000000000000, 0b0011111111100000000000000000000000000000000000000000000000000000)
    assert machine._reg_zfloat_result == 0b0011111111100000000000000000000000000000000000000000000000000000

def test_softfloat_f64muladd():
    machine = DummyMachine()
    supportcode.softfloat_f64muladd(machine, 0, 0b0011111111110000000000000000000000000000000000000000000000000000, 0b0011111111100000000000000000000000000000000000000000000000000000, 0b0011111111110000000000000000000000000000000000000000000000000000)
    assert machine._reg_zfloat_result == 0b0011111111111000000000000000000000000000000000000000000000000000

def test_softfloat_f64sqrt():
    machine = DummyMachine()
    supportcode.softfloat_f64sqrt(machine, 0, 0b0100000000010000000000000000000000000000000000000000000000000000)
    assert machine._reg_zfloat_result == 0b0100000000000000000000000000000000000000000000000000000000000000

def test_softfloat_f64sub():
    machine = DummyMachine()
    supportcode.softfloat_f64sub(machine, 0, 0b0011111111110000000000000000000000000000000000000000000000000000, 0b0011111111100000000000000000000000000000000000000000000000000000)
    assert machine._reg_zfloat_result == 0b0011111111100000000000000000000000000000000000000000000000000000
    
def test_softfloat_f16tof32():
    machine = DummyMachine()
    supportcode.softfloat_f16tof32(machine, 0, 0b0011100000000000)
    assert machine._reg_zfloat_result == 0b00111111000000000000000000000000
    
def test_softfloat_f32tof16():
    machine = DummyMachine()
    supportcode.softfloat_f32tof16(machine, 0, 0b00111111000000000000000000000000)
    assert machine._reg_zfloat_result == 0b0011100000000000
    
def test_softfloat_f16tof64():
    machine = DummyMachine()
    supportcode.softfloat_f16tof64(machine, 0, 0b0011100000000000)
    assert machine._reg_zfloat_result == 0b0011111111100000000000000000000000000000000000000000000000000000
    
def test_softfloat_f64tof16():
    machine = DummyMachine()
    supportcode.softfloat_f64tof16(machine, 0, 0b0011111111100000000000000000000000000000000000000000000000000000)
    assert machine._reg_zfloat_result == 0b0011100000000000
    
def test_softfloat_f32tof64():
    machine = DummyMachine()
    supportcode.softfloat_f32tof64(machine, 0, 0b00111111000000000000000000000000)
    assert machine._reg_zfloat_result == 0b0011111111100000000000000000000000000000000000000000000000000000
    
def test_softfloat_f64tof32():
    machine = DummyMachine()
    supportcode.softfloat_f64tof32(machine, 0, 0b0011111111100000000000000000000000000000000000000000000000000000)
    assert machine._reg_zfloat_result == 0b00111111000000000000000000000000

def test_softfloat_f16toi32():
    machine = DummyMachine()
    supportcode.softfloat_f16toi32(machine, 0, 0b1100010100000000)
    assert machine._reg_zfloat_result == 0b11111111111111111111111111111011

def test_softfloat_f16toi64():
    machine = DummyMachine()
    supportcode.softfloat_f16toi64(machine, 0, 0b1100010100000000)
    assert machine._reg_zfloat_result == 0b1111111111111111111111111111111111111111111111111111111111111011

def test_softfloat_f16toui32():
    machine = DummyMachine()
    supportcode.softfloat_f16toui32(machine, 0, 0b0100010100000000)
    assert machine._reg_zfloat_result == 0b00000000000000000000000000000101

def test_softfloat_f16toui64():
    machine = DummyMachine()
    supportcode.softfloat_f16toui64(machine, 0, 0b0100010100000000)
    assert machine._reg_zfloat_result == 0b0000000000000000000000000000000000000000000000000000000000000101

def test_softfloat_f32toi32():
    machine = DummyMachine()
    supportcode.softfloat_f32toi32(machine, 0, 0b11000000101000000000000000000000)
    assert machine._reg_zfloat_result == 0b11111111111111111111111111111011

def test_softfloat_f32toi64():
    machine = DummyMachine()
    supportcode.softfloat_f32toi64(machine, 0, 0b11000000101000000000000000000000)
    assert machine._reg_zfloat_result == 0b1111111111111111111111111111111111111111111111111111111111111011

def test_softfloat_f32toui32():
    machine = DummyMachine()
    supportcode.softfloat_f32toui32(machine, 0, 0b01000000101000000000000000000000)
    assert machine._reg_zfloat_result == 0b00000000000000000000000000000101

def test_softfloat_f32toui64():
    machine = DummyMachine()
    supportcode.softfloat_f32toui64(machine, 0, 0b01000000101000000000000000000000)
    assert machine._reg_zfloat_result == 0b0000000000000000000000000000000000000000000000000000000000000101

def test_softfloat_f64toi32():
    machine = DummyMachine()
    supportcode.softfloat_f64toi32(machine, 0, r_ulonglong(0b1100000000010100000000000000000000000000000000000000000000000000))
    assert machine._reg_zfloat_result == 0b11111111111111111111111111111011

def test_softfloat_f64toi64():
    machine = DummyMachine()
    supportcode.softfloat_f64toi64(machine, 0, r_ulonglong(0b1100000000010100000000000000000000000000000000000000000000000000))
    assert machine._reg_zfloat_result == 0b1111111111111111111111111111111111111111111111111111111111111011

def test_softfloat_f64toui32():
    machine = DummyMachine()
    supportcode.softfloat_f64toui32(machine, 0, 0b0100000000010100000000000000000000000000000000000000000000000000)
    assert machine._reg_zfloat_result == 0b00000000000000000000000000000101

def test_softfloat_f64toui64():
    machine = DummyMachine()
    supportcode.softfloat_f64toui64(machine, 0, 0b0100000000010100000000000000000000000000000000000000000000000000)
    assert machine._reg_zfloat_result == 0b0000000000000000000000000000000000000000000000000000000000000101

def test_softfloat_i32tof16():
    machine = DummyMachine()
    supportcode.softfloat_i32tof16(machine, 0, 0b11111111111111111111111111111011)
    assert machine._reg_zfloat_result == 0b1100010100000000

def test_softfloat_i32tof32():
    machine = DummyMachine()
    supportcode.softfloat_i32tof32(machine, 0, 0b11111111111111111111111111111011)
    assert machine._reg_zfloat_result == 0b11000000101000000000000000000000

def test_softfloat_i32tof64():
    machine = DummyMachine()
    supportcode.softfloat_i32tof64(machine, 0, 0b11111111111111111111111111111011)
    assert machine._reg_zfloat_result == 0b1100000000010100000000000000000000000000000000000000000000000000

def test_softfloat_i64tof16():
    machine = DummyMachine()
    supportcode.softfloat_i64tof16(machine, 0, r_ulonglong(0b1111111111111111111111111111111111111111111111111111111111111011))
    assert machine._reg_zfloat_result == 0b1100010100000000

def test_softfloat_i64tof32():
    machine = DummyMachine()
    supportcode.softfloat_i64tof32(machine, 0, r_ulonglong(0b1111111111111111111111111111111111111111111111111111111111111011))
    assert machine._reg_zfloat_result == 0b11000000101000000000000000000000

def test_softfloat_i64tof64():
    machine = DummyMachine()
    supportcode.softfloat_i64tof64(machine, 0, r_ulonglong(0b1111111111111111111111111111111111111111111111111111111111111011))
    assert machine._reg_zfloat_result == 0b1100000000010100000000000000000000000000000000000000000000000000

def test_softfloat_ui32tof16():
    machine = DummyMachine()
    supportcode.softfloat_ui32tof16(machine, 0, 0b00000000000000000000000000000101)
    assert machine._reg_zfloat_result == 0b0100010100000000

def test_softfloat_ui32tof32():
    machine = DummyMachine()
    supportcode.softfloat_ui32tof32(machine, 0, 0b00000000000000000000000000000101)
    assert machine._reg_zfloat_result == 0b01000000101000000000000000000000

def test_softfloat_ui32tof64():
    machine = DummyMachine()
    supportcode.softfloat_ui32tof64(machine, 0, 0b00000000000000000000000000000101)
    assert machine._reg_zfloat_result == 0b0100000000010100000000000000000000000000000000000000000000000000

def test_softfloat_ui64tof16():
    machine = DummyMachine()
    supportcode.softfloat_ui64tof16(machine, 0, 0b0000000000000000000000000000000000000000000000000000000000000101)
    assert machine._reg_zfloat_result == 0b0100010100000000

def test_softfloat_ui64tof32():
    machine = DummyMachine()
    supportcode.softfloat_ui64tof32(machine, 0, 0b0000000000000000000000000000000000000000000000000000000000000101)
    assert machine._reg_zfloat_result == 0b01000000101000000000000000000000

def test_softfloat_ui64tof64():
    machine = DummyMachine()
    supportcode.softfloat_ui64tof64(machine, 0, 0b0000000000000000000000000000000000000000000000000000000000000101)
    assert machine._reg_zfloat_result == 0b0100000000010100000000000000000000000000000000000000000000000000

#section for test file for sparse bitvectors


def test_sparse_add_int():
    v = SparseBitVector(100, r_uint(0b111))
    res = v.add_int(Integer.fromint(0b1))
    assert res.touint() == 0b1000

def test_sparse_read_bit():
    v = SparseBitVector(100, r_uint(0b10101))
    assert v.read_bit(4) == True
    assert v.read_bit(3) == False
    assert v.read_bit(2) == True
    assert v.read_bit(1) == False
    assert v.read_bit(0) == True
    assert v.read_bit(65) == False
    assert v.read_bit(99) == False
    with pytest.raises(AssertionError):
        v.read_bit(100) 

def test_sparse_vector_shift():
    v = SparseBitVector(100, 0b10001101)

    res = v.rshift(5)
    assert res.size() == 100
    assert res.toint() == 0b00000100

    res = v.rshift(100)
    assert res.size() == 100
    assert res.toint() == 0
    
    res = v.rshift(65)
    assert res.size() == 100
    assert res.toint() == 0

def test_sparse_arith_shiftr():
    v = SparseBitVector(100, 0b00101101)
    res = v.arith_rshift(3)
    assert res.size() == 100
    assert res.toint() == 0b101

    v = SparseBitVector(100, 0b1000100)
    res = v.arith_rshift(6)
    assert res.size() == 100
    assert res.toint() == 0b1

def test_sparse_vector_shift_bits():
    v = SparseBitVector(100, 0b10001101)
    res = v.rshift_bits(SparseBitVector(100, 5))
    assert res.size() == 100
    assert res.toint() == 0b00000100

    v = SparseBitVector(100, 0b10001101)
    res = v.rshift_bits(SparseBitVector(100, 65))
    assert res.size() == 100
    assert res.toint() == 0

def test_sparse_bv_bitwise():
    v1 = SparseBitVector(100, 0b11110000)
    v2 = SparseBitVector(100, 0b11001100)
    res = v1.and_(v2)
    assert res.toint() == 0b11110000 & 0b11001100
    res = v1.or_(v2)
    assert res.toint() == 0b11110000 | 0b11001100
    res = v1.xor(v2)
    assert res.toint() == 0b11110000 ^ 0b11001100

def test_sparse_zero_extend():
    # XXX Should I test it with support code?
    v = SparseBitVector(65, 0b0)
    res = v.zero_extend(100)
    assert res.size() == 100
    assert res.toint() == 0

    v = SparseBitVector(100, 0b00)
    res = v.zero_extend(100)
    assert res.size() == 100
    assert res.toint() == 0

    v = SparseBitVector(65, 0b1)
    res = v.zero_extend(100)
    assert res.size() == 100
    assert res.toint() == 0b00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001

    v = SparseBitVector(65, 0b11)
    res = v.zero_extend(100)
    assert res.size() == 100
    assert res.toint() == 0b0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011

def test_sparse_sign_extend():
    # XXX Should I test it with support code?
    v = SparseBitVector(65, 0b0)
    res = v.sign_extend(100)
    assert res.size() == 100
    assert res.toint() == 0

    v = SparseBitVector(100, 0b00)
    res = v.sign_extend(100)
    assert res.size() == 100
    assert res.toint() == 0

    v = SparseBitVector(65, 0b1)
    res = v.sign_extend(100)
    assert res.size() == 100
    assert res.toint() == 0b00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001

    v = SparseBitVector(65, 0b11)
    res = v.sign_extend(100)
    assert res.size() == 100
    assert res.toint() == 0b0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011


def test_sparse_vector_subrange():
    # XXX Regression bug and sail implementation
    v = SparseBitVector(100, 0b111)
    r = v.subrange(3, 2)
    assert r.size() == 2
    assert r.toint() == 1
    assert isinstance(r, bitvector.SmallBitVector)

    # BUG Doesnt work on 64 width
    # v = SparseBitVector(65, 0b111)
    # r = v.subrange(63, 0)
    # assert r.size() == 64
    # assert r.toint() == 0b111
    # assert isinstance(r, bitvector.SmallBitVector)

    # v = SparseBitVector(100, 0b101010101)
    # r = v.subrange(65, 2)
    # assert r.size() == 64
    # assert r.toint() == 0b101010
    # assert isinstance(r, bitvector.SmallBitVector)

    v = SparseBitVector(100, 0b101010101)
    r = v.subrange(5, 0)
    assert r.size() == 6
    assert r.toint() == 0b10101
    assert isinstance(r, bitvector.SmallBitVector)

    v = SparseBitVector(100, 0b101010101)
    r = v.subrange(65, 0)
    assert r.size() == 66
    assert r.toint() == 0b101010101
    assert isinstance(r, bitvector.SparseBitVector)

    v = SparseBitVector(100, 0b101010101)
    r = v.subrange(65, 3)
    assert r.size() == 63
    assert r.toint() == 0b101010
    assert isinstance(r, bitvector.SmallBitVector)

    v = SparseBitVector(100, 0b101010101)
    r = v.subrange(65, 1)
    assert r.size() == 65
    assert r.toint() == 0b10101010
    assert isinstance(r, bitvector.SparseBitVector)

    v = SparseBitVector(100, 0b101010101)
    r = v.subrange(99, 0)
    assert r.size() == 100
    assert r.toint() == 0b101010101
    assert isinstance(r, bitvector.SparseBitVector)

def test_sparse_vector_update():
    v = SparseBitVector(100, 1)
    res = v.update_bit(2, 1)
    assert res.size() == 100
    assert res.toint() == 0b101
    
    v = SparseBitVector(65, r_uint(1))
    res = v.update_bit(1, 0)
    assert res.size() == 65
    assert res.toint() == 0b1

    v = SparseBitVector(65, r_uint(2))
    res = v.update_bit(0, 0)
    assert res.size() == 65
    assert res.toint() == 0b10

    v = SparseBitVector(65, r_uint(0))
    res = v.update_bit(1, 1)
    assert res.size() == 65
    assert res.tolong() == 0b10

    v = SparseBitVector(100, 1)
    res = v.update_bit(0, 1)
    assert res.size() == 100
    assert res.toint() == 0b1

    v = SparseBitVector(100, 0b11)
    res = v.update_bit(2, 0)
    assert res.size() == 100
    assert res.toint() == 0b011

    v = SparseBitVector(100, 0b111)
    res = v.update_bit(1, 0)
    assert res.size() == 100
    assert res.toint() == 0b101

    v = SparseBitVector(100, 0b111)
    res = v.update_bit(65, 0)
    assert res.size() == 100
    assert res.toint() == 0b111
    assert isinstance(res, bitvector.GenericBitVector)

def test_sparse_signed():
    # XXX Machine?
    v = SparseBitVector(65, 0b0)
    assert v.signed().toint() == 0 
    assert isinstance(v.signed(), SmallInteger)

def test_sparse_unsigned():
    v = SparseBitVector(100, 0b10001101)
    assert v.unsigned().tolong() == 0b10001101

    v = SparseBitVector(100, r_uint(-1))
    assert v.unsigned().tolong() == (1<<64)-1

def test_sparse_truncate():
    res = SparseBitVector(100, 0b1011010100).truncate(2)
    assert isinstance(res, bitvector.SmallBitVector)
    assert res.size() == 2
    assert res.touint() == 0b00
    res = SparseBitVector(100, 0b1011010100).truncate(6)
    assert res.size() == 6
    assert res.touint() == 0b010100
    res = SparseBitVector(100, 0b1011010100).truncate(100)
    assert isinstance(res, bitvector.SparseBitVector)
    assert res.touint() == 0b1011010100

def test_sparse_eq():
    assert SparseBitVector(100, -12331).eq(SparseBitVector(100, -12331))
    assert not SparseBitVector(100, -12331).eq(SparseBitVector(100, 12331))
    assert SparseBitVector(100, 0b10111).eq(bitvector.GenericBitVector(100, rbigint.fromlong(0b10111)))

def test_sparse_lshift():
    v = SparseBitVector(100, 0b10001101)
    res = v.lshift(5)
    assert res.size() == 100
    assert res.toint() == 0b1000110100000
    assert isinstance(res, SparseBitVector)

    v = SparseBitVector(65, 1)
    res = v.lshift(65)
    assert res.size() == 65
    assert res.tolong() == 0
    assert isinstance(res, bitvector.GenericBitVector)
    
    v = SparseBitVector(100, 0b0010000000000000000000000000000000000000000000000000000000000000)
    res = v.lshift(1)
    assert res.size() == 100
    assert res.toint() == 0b00100000000000000000000000000000000000000000000000000000000000000


    v = SparseBitVector(100, r_uint(1) << 63)
    res = v.lshift(1)
    assert res.size() == 100
    assert isinstance(res, bitvector.GenericBitVector)
    
def test_sparse_check_carry():
    v = SparseBitVector(100, r_uint(0xffffffffffffffff))
    assert v.check_carry(r_uint(0b1)) == 1
    v = SparseBitVector(100, r_uint(0xfffffffffffffffe))
    assert v.check_carry(r_uint(0b1)) == 0
    v = SparseBitVector(100, r_uint(0xfffffffffffffffe))
    assert v.check_carry(r_uint(0b10)) == 1
    v = SparseBitVector(100, r_uint(0xffffffffffffffee))
    assert v.check_carry(r_uint(0xffffffff)) == 1
    v = SparseBitVector(100, r_uint(0xffffffffffffffee))
    assert v.check_carry(r_uint(0x1)) == 0
    v = SparseBitVector(100, r_uint(0x0))
    assert v.check_carry(r_uint(0x1)) == 0


def test_sparse_add_int():
    for c in bi, si:
        assert SparseBitVector(6000, 0b11).add_int(c(0b111111111)).touint() == 0b11 + 0b111111111
        assert SparseBitVector(6000, r_uint(0xfffffffffffffffe)).add_int(c(0b1)).tolong() == 0xfffffffffffffffe + 1
        assert isinstance (SparseBitVector(100, r_uint(0xffffffffffffffff)).add_int(c(0b1)), bitvector.GenericBitVector)
        assert isinstance (SparseBitVector(100, r_uint(0xfffffffffffffffee)).add_int(c(0xfff)), bitvector.GenericBitVector)

def test_sparse_add_bits():
    for c in SparseBitVector, gbv:
        assert SparseBitVector(100, 0b11).add_bits(c(100, 0b111111111)).touint() == 0b11 + 0b111111111
        assert SparseBitVector(100, r_uint(0xfffffffffffffffe)).add_bits(SparseBitVector(100, 0b1)).tolong() == 0xfffffffffffffffe + 1
    assert isinstance(SparseBitVector(65, r_uint(0xffffffffffffffff)).add_bits(SparseBitVector(65,0b1)), bitvector.GenericBitVector)

def test_sparse_sub_bits():
    for c in gbv, SparseBitVector:
        assert (SparseBitVector(100, (0b0)).sub_bits(c(100, r_uint(0b1))), bitvector.GenericBitVector)
        assert SparseBitVector(100, 0b0).sub_bits(c(100, 0b1)).tolong() == -1 % (2 ** 100)
        assert SparseBitVector(100, r_uint(0xffffffffffffffff)).sub_bits(c(100, 0b1)).tolong() == 0xffffffffffffffff - 1

def test_sparse_sub_int():
    for c in bi, si:
        assert SparseBitVector(100, 0b0).sub_int(c(0b1)).tolong() == -1 % (2 ** 100)
        assert SparseBitVector(6000, r_uint(0xffffffffffffffff)).sub_int(c(0b1)).tolong() == 0xffffffffffffffff -1 
        assert SparseBitVector(68, 4).sub_int(c(9)).tolong() == -5 % (2 ** 68)
        assert SparseBitVector(100, r_uint(18446744073709486081)).sub_int(c(-65535)).tolong() == 18446744073709551616
        assert SparseBitVector(68, 0b0).sub_int(c(0b1)).tolong() == -1 % (2 **68)
    assert isinstance(SparseBitVector(6000, 0b11).sub_int(si(0b11)), SparseBitVector)
    assert isinstance(SparseBitVector(6000, r_uint(0xffffffffffffffff)).sub_int(bi(0b1)), bitvector.GenericBitVector)

        
@given(strategies.data())
def test_sparse_hypothesis_sub_int(data):
    value1 = data.draw(strategies.integers(0, 2**64 - 1))
    value2 = data.draw(strategies.integers(MININT, sys.maxint))
    ans = value1 - value2
    for c in bi, si:
        assert SparseBitVector(100, r_uint(value1)).sub_int(c(value2)).tolong() == ans % (2 ** 100)

@given(strategies.data())
def test_sparse_hypothesis_sub_bits(data):
    value1 = data.draw(strategies.integers(0, 2**64 - 1))
    value2 = data.draw(strategies.integers(0, sys.maxint))
    ans = value1 - value2
    for c in gbv, SparseBitVector:
        assert SparseBitVector(100, r_uint(value1)).sub_bits(c(100, r_uint(value2))).tolong() == ans % (2 ** 100)

@given(strategies.data())
def test_sparse_hypothesis_add_bits(data):
    value1 = data.draw(strategies.integers(0, 2**64 - 1))
    value2 = data.draw(strategies.integers(0, sys.maxint))
    ans = value1 + value2
    for c in gbv, SparseBitVector:
        assert SparseBitVector(100, r_uint(value1)).add_bits(c(100, r_uint(value2))).tolong() == ans 

        
@given(strategies.data())
def test_sparse_hypothesis_add_int(data):
    value1 = data.draw(strategies.integers(0, 2**64 - 1))
    value2 = data.draw(strategies.integers(MININT, sys.maxint))
    ans = value1 + value2
    for c in bi, si:
        if ans >= 0:
            assert SparseBitVector(100, r_uint(value1)).add_int(c(value2)).tolong() == ans 
        assert SparseBitVector(100, r_uint(value1)).add_int(c(value2)).tolong() == ans % (2 ** 100)

@given(strategies.data())
def test_sparse_hypothesis_truncate(data):
    bitwidth = data.draw(strategies.integers(65, 10000))
    truncatewidth = data.draw(strategies.integers(1, bitwidth))
    value = data.draw(strategies.integers(0, 2**64 - 1))
    as_bit_string = bin(value)[2:]
    bv = SparseBitVector(bitwidth, r_uint(value))
    res = bv.truncate(truncatewidth)
    assert bin(bv.tolong())[2:].rjust(bitwidth, '0')[-truncatewidth:] == bin(res.tolong())[2:].rjust(truncatewidth, '0')

@given(strategies.data())
def test_sparse_hypothesis_vector_subrange(data):
    bitwidth = data.draw(strategies.integers(65, 10000))
    # TODO m- n + 1 = 64 wont work
    lower = data.draw(strategies.integers(0, 62))
    upper = data.draw(strategies.integers(lower, 62))
    value = data.draw(strategies.integers(0, 2**63 - 1))
    as_bit_string = bin(value)[2:]
    assert len(as_bit_string) <= bitwidth
    as_bit_string = as_bit_string.rjust(bitwidth, '0')[::-1]
    correct_res = as_bit_string[lower:upper+1] # sail is inclusive
    correct_res_as_int = int(correct_res[::-1], 2)

    # now do the sail computation
    v = SparseBitVector(bitwidth, value)
    vres = v.subrange(upper, lower)
    assert vres.tobigint().tolong() == correct_res_as_int

@settings(deadline=1000)
@given(strategies.data())
def test_sparse_hypothesis_sign_extend(data):
    bitwidth = data.draw(strategies.integers(65, 10000))
    target_bitwidth = bitwidth + data.draw(strategies.integers(1, 100))
    value = data.draw(strategies.integers(0, 2**64 - 1))
    bv = SparseBitVector(bitwidth, r_uint(value))
    res = bv.sign_extend(target_bitwidth)
    print bitwidth, target_bitwidth, value, bv, res, bv.signed().tobigint(), res.signed().tobigint()
    assert bv.signed().tobigint().tolong() == res.signed().tobigint().tolong()

@settings(deadline=1000)
@given(strategies.data())
def test_sparse_hypothesis_zero_extend(data):
    bitwidth = data.draw(strategies.integers(65, 10000))
    target_bitwidth = bitwidth + data.draw(strategies.integers(1, 100))
    value = data.draw(strategies.integers(0, 2**64 - 1))
    bv = SparseBitVector(bitwidth, r_uint(value))
    res = bv.zero_extend(target_bitwidth)
    print bitwidth, target_bitwidth, value, bv, res, bv.signed().tobigint(), res.signed().tobigint()
    assert bv.signed().tobigint().tolong() == res.signed().tobigint().tolong()

@given(strategies.data())
@settings(deadline = None)
def test_sparse_hypothesis_replicate(data):
    bitwidth = data.draw(strategies.integers(65, 10000))
    repeats = data.draw(strategies.integers(1, 10))
    value = data.draw(strategies.integers(0, 2 **64 - 1))
    bv = SparseBitVector(bitwidth, r_uint(value))
    res = bv.replicate(repeats)
    ans_as_int = bin(value)
    formatted_value = str(ans_as_int)[2:]
    leading_zero = (str(0)* (bitwidth - len(formatted_value)) + formatted_value)
    assert len(leading_zero) == bitwidth
    ans = str(leading_zero) * repeats
    assert res.tolong() == int(ans, 2) 


@given(strategies.data())
def test_sparse_hypothesis_eq(data):
    bitwidth = data.draw(strategies.integers(65,10000))
    value = data.draw(strategies.integers(0, 2**64- 1))
    if not data.draw(strategies.booleans()):
        bv = SparseBitVector(bitwidth, r_uint(value))
    else:
        bv = gbv(bitwidth, r_uint(value))
    v = SparseBitVector(bitwidth, r_uint(value))
    assert v.eq(bv)

@given(strategies.data())
def test_sparse_hypothesis_update_bit(data):
    bitwidth = data.draw(strategies.integers(65,10000))
    value = data.draw(strategies.integers(0, 2**64- 1))
    pos = data.draw(strategies.integers(0, bitwidth -1))
    bit = data.draw(strategies.integers(0, 1))
    v = SparseBitVector(bitwidth, r_uint(value))
    formatted_value = str(bin(value))[2:]
    value = formatted_value.rjust(bitwidth, '0')[::-1]
    assert len(value) == bitwidth
    if pos == 0: 
        value = str(bit) + value[1:]
    elif pos == bitwidth - 1:
        value = value[:pos] + str(bit)
    else:
        value = value[:pos] + str(bit) + value[pos + 1:]
    res = v.update_bit(pos, bit)
    assert res.tolong() == int(value[::-1],2)

@given(strategies.data())
def test_sparse_hypothesis_read_bit(data):
    bitwidth = data.draw(strategies.integers(65,10000))
    value = data.draw(strategies.integers(0, 2**64- 1))
    pos = data.draw(strategies.integers(0, bitwidth -1))
    value_as_str = str(bin(value))
    formatted_value = value_as_str[2:]
    v = SparseBitVector(bitwidth, r_uint(value))
    assert v.read_bit(pos) == int(formatted_value.rjust(bitwidth, '0')[::-1][pos])

@given(strategies.data())
def test_sparse_hypothesis_op(data):
    bitwidth = data.draw(strategies.integers(65,10000))
    value1 = data.draw(strategies.integers(0, 2**64- 1))
    value2 = data.draw(strategies.integers(0, 2**64- 1))
    for c1 in SparseBitVector, gbv:
        for c2 in SparseBitVector, gbv:
            assert c1(bitwidth, r_uint(value1)).xor(c2(bitwidth, r_uint(value2))).tolong() == (value1 ^ value2)
            assert c1(bitwidth, r_uint(value1)).or_(c2(bitwidth, r_uint(value2))).tolong() == (value1 | value2)
            assert c1(bitwidth, r_uint(value1)).and_(c2(bitwidth, r_uint(value2))).tolong() == (value1 & value2)


@given(strategies.data())
def test_sparse_hypothesis_invert(data):
    bitwidth = data.draw(strategies.integers(65,10000))
    value = data.draw(strategies.integers(0, 2**64- 1))
    v = SparseBitVector(bitwidth, r_uint(value))
    value_as_str = str(bin(value))
    formatted_value = value_as_str[2:]
    filled = "0b" + formatted_value.rjust(bitwidth, '0')
    inverse_s = ~int(filled,2) % (2 ** bitwidth)
    assert v.invert().tolong() == inverse_s

@given(strategies.data())
def test_sparse_hypothesis_unsigned(data):
    bitwidth = data.draw(strategies.integers(65,10000))
    value = data.draw(strategies.integers(0, 2**64- 1))
    v = SparseBitVector(bitwidth, r_uint(value))
    value_as_str = str(bin(value))
    formatted_value = value_as_str[2:]
    filled = formatted_value.rjust(bitwidth, '0')
    assert v.unsigned().tolong() == int(filled, 2)

@given(strategies.data())
def test_sparse_hypothesis_signed(data):
    bitwidth = data.draw(strategies.integers(65,10000))
    value = data.draw(strategies.integers(-(2**63), (2**63)- 1))
    v = SparseBitVector(bitwidth, r_uint(value))
    # it could never be negative when interpret as signed
    assert v.signed().tolong() >= 0
    assert v.signed().tolong() == r_uint(value)


@given(strategies.data())
def test_sparse_hypothesis_lshift(data):
    bitwidth = data.draw(strategies.integers(65,10000))
    value = data.draw(strategies.integers(0, 2**64- 1))
    v = SparseBitVector(bitwidth, r_uint(value))
    shift = data.draw(strategies.integers(0, bitwidth))
    res = v.lshift(shift).tolong()
    mask = ''
    assert res == (value << shift) & ((1 << bitwidth) - 1) 

@given(strategies.data())
def test_sparse_hypothesis_lshift_bits(data):
    bitwidth = data.draw(strategies.integers(65,10000))
    value1 = data.draw(strategies.integers(0, 2**64- 1))
    value2 = data.draw(strategies.integers(0, bitwidth))
    v1 = SparseBitVector(bitwidth, r_uint(value1))
    v2 = SparseBitVector(bitwidth, r_uint(value2))
    res = v1.lshift_bits(v2).tolong()
    mask = ''
    assert res == (value1 << value2) & ((1 << bitwidth) - 1) 

@given(strategies.data())
def test_sparse_hypothesis_rshift(data):
    bitwidth = data.draw(strategies.integers(65,10000))
    value = data.draw(strategies.integers(0, 2**64- 1))
    v = SparseBitVector(bitwidth, r_uint(value))
    shift = data.draw(strategies.integers(0, bitwidth))
    res = v.rshift(shift).tolong()
    assert res == (value >> shift)

@given(strategies.data())
def test_sparse_hypothesis_rshift_bits(data):
    bitwidth = data.draw(strategies.integers(65,10000))
    value1 = data.draw(strategies.integers(0, 2**64- 1))
    value2 = data.draw(strategies.integers(0, bitwidth))
    v1 = SparseBitVector(bitwidth, r_uint(value1))
    v2 = SparseBitVector(bitwidth, r_uint(value2))
    res = v1.rshift_bits(v2).tolong()
    mask = ''
    assert res == (value1 >> value2) 

@given(strategies.data())
def test_sparse_arith_shiftr_hypothesis(data):
    size = data.draw(strategies.integers(65, 5000))
    value = data.draw(strategies.integers(0, 2**size-1))
    v = bitvector.SparseBitVector(size, r_uint(value))
    shift = data.draw(strategies.integers(0, size+10))
    res = v.arith_rshift(shift)
    intres = v.signed().tobigint().tolong() >> shift
    assert res.tobigint().tolong() == intres & ((1 << size) - 1)
# def test_smallbitvector():
#     x = SmallBitVector(4, r_uint(4))
#     # 1011 --> 1100 --> 1101
#     assert x.read_bit(0) == 0
#     assert x.read_bit(1) == 0
#     assert x.read_bit(2) == 1
#     assert x.read_bit(3) == 0
#     x = x.arith_rshift(1)
#     assert x.read_bit(0) == 0
#     assert x.read_bit(1) == 1
#     assert x.read_bit(2) == 0
#     assert x.read_bit(3) == 0
#     x = SmallBitVector(4, r_uint(-4))
#     # 1100 --> 1011 --> 1100
#     assert x.read_bit(0) == 0
#     assert x.read_bit(1) == 0
#     assert x.read_bit(2) == 1
#     assert x.read_bit(3) == 1
#     x = x.arith_rshift(1)
#     assert x.read_bit(0) == 0
#     assert x.read_bit(1) == 1
#     assert x.read_bit(2) == 1
#     assert x.read_bit(3) == 1
#     assert x.eq(SmallBitVector(4, r_uint(-2))) == True
#     x = SmallBitVector(4, r_uint(3))
#     x = x.arith_rshift(1)
#     assert x.read_bit(0) == 1
#     assert x.read_bit(1) == 0
#     assert x.read_bit(2) == 0
#     assert x.read_bit(3) == 0
#     x = SmallBitVector(4, r_uint(-3))
#     # 1011 --> 1100 --> 1101
#     assert x.read_bit(0) == 1
#     assert x.read_bit(1) == 0
#     assert x.read_bit(2) == 1
#     assert x.read_bit(3) == 1
#     x = x.arith_rshift(1)
#     assert x.read_bit(0) == 0
#     assert x.read_bit(1) == 1
#     assert x.read_bit(2) == 1
#     assert x.read_bit(3) == 1
#     assert x.eq(SmallBitVector(4, r_uint(-2))) == True
#     x = SmallBitVector(6, r_uint(-7))
#     # 100111 --> 111000 --> 111001
#     assert x.read_bit(0) == 1
#     assert x.read_bit(1) == 0
#     assert x.read_bit(2) == 0
#     assert x.read_bit(3) == 1
#     assert x.read_bit(4) == 1
#     assert x.read_bit(5) == 1
#     x = x.arith_rshift(2)
#     assert x.read_bit(0) == 0
#     assert x.read_bit(1) == 1
#     assert x.read_bit(2) == 1
#     assert x.read_bit(3) == 1
#     assert x.read_bit(4) == 1
#     assert x.read_bit(5) == 1
#     assert x.eq(SmallBitVector(6, r_uint(-2)))
#     x = SmallBitVector(4, r_uint(-1))
#     x = x.arith_rshift(5)
#     assert x.read_bit(0) == 1
#     assert x.read_bit(1) == 1
#     assert x.read_bit(2) == 1
#     assert x.read_bit(3) == 1
#     x = SmallBitVector(4, r_uint(-1))
#     # 1001 --> 1110 --> 1111
#     x = x.arith_rshift(2)
#     assert x.read_bit(0) == 1
#     assert x.read_bit(1) == 1
#     assert x.read_bit(2) == 1
#     assert x.read_bit(3) == 1
#     x = SmallBitVector(4, r_uint(0))
#     assert x.read_bit(0) == 0
#     assert x.read_bit(1) == 0
#     assert x.read_bit(2) == 0
#     assert x.read_bit(3) == 0
#     x = SmallBitVector(4, r_uint(17))
#     assert x.read_bit(0) == 1
#     assert x.read_bit(1) == 0
#     assert x.read_bit(2) == 0
#     assert x.read_bit(3) == 0 
#     a = -6
#     # assert str(bin(abs(a)))[2:] == "111"
#     val_bin = abs(a)
#     length = len(bin(val_bin))-2
#     val_bin = (val_bin ^ ((1 << len(bin(val_bin)[2:])) -1)) + 1
#     val_str = "0"*(length-len(bin(val_bin)[2:]))+bin(val_bin)[2:]
#     assert val_str == "010"

def test_genericbitvector():
    # x = GenericBitVector(4, rbigint.fromint(4))
    # assert x.read_bit(0) == 0
    # assert x.read_bit(1) == 0
    # assert x.read_bit(2) == 1
    # assert x.read_bit(3) == 0
    # # x = GenericBitVector(3, rbigint.fromint(-2))
    # # x = x.arith_shiftr(2)
    # # assert x.read_bit(0) == 1
    # x = GenericBitVector(3, rbigint.fromint(-2))
    # # 1010 --> 1101 --> 1110
    # assert x.read_bit(0) == 0
    # assert x.read_bit(1) == 1
    # assert x.read_bit(2) == 1
    x = GenericBitVector(3, rbigint.fromint(-1))
    assert x.rval.and_(rbigint.fromint(1)).toint() == 1
    assert x.rval.and_(rbigint.fromint(1).lshift(1)).rshift(1).toint() == 1
    assert x.rval.and_(rbigint.fromint(1).lshift(2)).rshift(2).toint() == 1

def test_slice_int_smallint():
    x = SmallBitVector(2, r_uint(-1))
    assert x.read_bit(0) == 1
    assert x.read_bit(1) == 1
    y = SmallInteger(0)
    out = y.set_slice_int(2, 0, x)
    assert out.val == 3



@given(strategies.integers(min_value = MININT, max_value = sys.maxint), strategies.integers(min_value = 0, max_value = 64), strategies.integers(min_value = 0, max_value = 65))
def test_arith_shiftr_smallbitvector_hypothesis(val, size, n):
    if val < 0:
        val_new = abs(val)
        length = len(bin(val_new))-2
        val_new = (val_new ^ ((1 << len(bin(val_new)[2:])) - 1)) + 1
        val_bin = "0"*(length-len(bin(val_new)[2:]))+bin(val_new)[2:]
    else:
        val_bin = bin(val)[2:]
    size_zero = size % (63 - len(val_bin)) if len(val_bin) < 63 else 0
    val_bin = "1"+"1"*size_zero+val_bin if val < 0 else ("0"+"0"*size_zero+val_bin if val > 0 else "0")
    x = SmallBitVector(len(val_bin), r_uint(val))
    x = x.arith_rshift(n)
    if val == 0:
        # assert x == SmallBitVector(len(val_bin), r_uint(val))
        assert x.toint() == 0
        # assert SmallBitVector(len(val_bin), r_uint(val)) == SmallBitVector(len(val_bin), r_uint(val))
    elif n > len(val_bin) and val > 0:
        for i in range(0, len(val_bin)):
            assert x.read_bit(i) == int(0)
    elif n > len(val_bin) and val < 0:
        for i in range(0, len(val_bin)):
            assert x.read_bit(i) == int(1)
    else:
        val_bin = val_bin[0]*n + val_bin[0:(len(val_bin)-n)]
        for i in range(0, len(val_bin)):
            assert x.read_bit(i) == int(val_bin[len(val_bin)-1-i])


@given(strategies.integers(min_value = MININT+1, max_value = sys.maxint), strategies.integers(min_value = -2**62, max_value = 2**63-1), strategies.integers(min_value = 0, max_value = 63), strategies.integers(min_value = 0, max_value = 64))
def test_set_slice_int_smallinteger_hypothesis(val_int, val_bv, start, bv_size):
    x = SmallInteger(val_int)
    if val_int < 0:
        val_int_new = abs(val_int)
        length_int = len(bin(val_int_new))-2
        val_int_new = (val_int_new ^ ((1 << len(bin(val_int_new)[2:])) - 1)) + 1
        val_int_bin = "1"+"0"*(length_int-len(bin(val_int_new)[2:]))+bin(val_int_new)[2:]
    else:
        val_int_bin = bin(val_int)[2:]
    if val_int == 0:
        assert val_int_bin == "0"
    if val_bv < 0:
        val_bv_new = abs(val_bv)
        length_bv = len(bin(val_bv_new))-2
        val_bv_new = (val_bv_new ^ ((1 << len(bin(val_bv_new)[2:])) - 1)) + 1
        val_bv_bin = "0"*(length_bv-len(bin(val_bv_new)[2:]))+bin(val_bv_new)[2:]
    else:
        val_bv_bin = bin(val_bv)[2:]
    if val_bv == 0:
        assert val_bv_bin == "0"
    size_zero = bv_size % (63 - len(val_bv_bin)) if len(val_bv_bin) < 63 else 0
    if bv_size == 0:
        assert size_zero == 0
    val_bv_bin = "1"+"1"*size_zero+val_bv_bin if val_bv < 0 else ("0"+"0"*size_zero+val_bv_bin if val_bv > 0 else "0")
    bv = SmallBitVector(len(val_bv_bin), r_uint(val_bv))
    x_after = x.set_slice_int(len(val_bv_bin), start, bv)
    # xxxxxx
    if len(val_int_bin)-start-len(val_bv_bin) >= 0:
        if start == 0:
            out_bin = val_int_bin[:len(val_int_bin)-start-len(val_bv_bin)] + val_bv_bin
        else:
            out_bin = val_int_bin[:len(val_int_bin)-start-len(val_bv_bin)] + val_bv_bin + val_int_bin[len(val_int_bin)-start:]
        if len(val_int_bin)-start-len(val_bv_bin) == 0:
            out_bin = "1" + out_bin if val_int < 0 else "0" + out_bin     
    else:
        if start == 0:
            out_bin = val_bv_bin
        else:
            if start <= len(val_int_bin):
                out_bin = val_bv_bin + val_int_bin[len(val_int_bin)-start:]
            elif val_int < 0:
                out_bin = val_bv_bin + "1"*(start - len(val_int_bin)) + val_int_bin
            else:
                out_bin = val_bv_bin + "0"*(start - len(val_int_bin)) + val_int_bin
        out_bin = "1" + out_bin if val_int < 0 else "0" + out_bin
    out_bin = "1" + out_bin if val_int < 0 else "0" + out_bin
    if val_int < 0:
        out_val = int(out_bin[len(out_bin)-1])
        for i in range(1, len(out_bin)):
            out_val = out_val + int(out_bin[len(out_bin)-1-i])*2**i
        out_val = out_val - 1
        out_val = -(out_val ^ ((1 << len(bin(out_val)[2:])) - 1))
    else:
        out_val = int(out_bin[len(out_bin)-1])
        for i in range(1, len(out_bin)):
            out_val = out_val + int(out_bin[len(out_bin)-1-i])*2**i
    if isinstance(x_after, SmallInteger):
        assert out_val == x_after.val
    else:
        assert str(out_val) == x_after.rval.str()

@given(strategies.integers(min_value = -2**66, max_value = 2**66-1), strategies.integers(min_value = -2**66, max_value = 2**66-1), strategies.integers(min_value = 0, max_value = 67), strategies.integers(min_value = 0, max_value = 64))
def test_set_slice_int_biginteger_hypothesis(val_int, val_bv, start, bv_size):
    x = BigInteger(rbigint.fromstr(str(val_int)))
    if val_int < 0:
        val_int_new = abs(val_int)
        length_int = len(bin(val_int_new))-2
        val_int_new = (val_int_new ^ ((1 << len(bin(val_int_new)[2:])) - 1)) + 1
        val_int_bin = "1"+"0"*(length_int-len(bin(val_int_new)[2:]))+bin(val_int_new)[2:]
    else:
        val_int_bin = bin(val_int)[2:]
    if val_int == 0:
        assert val_int_bin == "0"
    if val_bv < 0:
        val_bv_new = abs(val_bv)
        length_bv = len(bin(val_bv_new))-2
        val_bv_new = (val_bv_new ^ ((1 << len(bin(val_bv_new)[2:])) - 1)) + 1
        val_bv_bin = "0"*(length_bv-len(bin(val_bv_new)[2:]))+bin(val_bv_new)[2:]
    else:
        val_bv_bin = bin(val_bv)[2:]
    size_zero = bv_size % (63 - len(val_bv_bin)) if len(val_bv_bin) < 63 else 0
    val_bv_bin = "1"+"1"*size_zero+val_bv_bin if val_bv < 0 else ("0"+"0"*size_zero+val_bv_bin if val_bv > 0 else "0")
    if val_int == -1 and bv_size == 0:
        assert val_int_bin == "11"
    bv = GenericBitVector(len(val_bv_bin), rbigint.fromstr(str(val_bv)))
    x_after = x.set_slice_int(len(val_bv_bin), start, bv)
    # xxxxxx
    if len(val_int_bin)-start-len(val_bv_bin) >= 0:
        if start == 0:
            out_bin = val_int_bin[:len(val_int_bin)-start-len(val_bv_bin)] + val_bv_bin
        else:
            out_bin = val_int_bin[:len(val_int_bin)-start-len(val_bv_bin)] + val_bv_bin + val_int_bin[len(val_int_bin)-start:]
        if len(val_int_bin)-start-len(val_bv_bin) == 0:
            out_bin = "1" + out_bin if val_int < 0 else "0" + out_bin     
    else:
        if start == 0:
            out_bin = val_bv_bin
        else:
            if start <= len(val_int_bin):
                out_bin = val_bv_bin + val_int_bin[len(val_int_bin)-start:]
            elif val_int < 0:
                out_bin = val_bv_bin + "1"*(start - len(val_int_bin)) + val_int_bin
            else:
                out_bin = val_bv_bin + "0"*(start - len(val_int_bin)) + val_int_bin
        out_bin = "1" + out_bin if val_int < 0 else "0" + out_bin
    out_bin = "1" + out_bin if val_int < 0 else "0" + out_bin
    if val_int < 0:
        out_val = int(out_bin[len(out_bin)-1])
        for i in range(1, len(out_bin)):
            out_val = out_val + int(out_bin[len(out_bin)-1-i])*2**i
        out_val = out_val - 1
        out_val = -(out_val ^ ((1 << len(bin(out_val)[2:])) - 1))
    else:
        out_val = int(out_bin[len(out_bin)-1])
        for i in range(1, len(out_bin)):
            out_val = out_val + int(out_bin[len(out_bin)-1-i])*2**i
    assert str(out_val) == x_after.rval.str()


    