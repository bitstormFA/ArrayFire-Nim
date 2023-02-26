import ArrayFire_Nim
import unittest
import strutils
import std/strformat


test "backend test":
    let backends = available_backends()
    echo "available backends $1" % $backends
    for b in backends:
        echo "testing backend $1" % $b
        setBackend(b)
        let active = getActiveBackend()
        check active == b
    let bc = getBackendCount()
    echo &"number of available backends: {bc}"
    check bc > 0

test "matrix construction with specified matrix type":
    let m1d = afa(9, @[1, 2, 3, 4, 5, 6, 7, 8, 9], Dtype.F64)
    echo m1d
    let n1d = afa(@[1, 2, 3, 4, 5, 6, 7, 8, 9], Dtype.S16)
    check m1d.dims(0) == 9
    check m1d.ndims == 1
    check len(m1d) == 9
    check m1d.elements == n1d.elements
    check m1d.dtype == Dtype.F64
    check n1d.dtype == Dtype.S16

test "matrix construction without matrix type":
    let m1d = afa(9, @[1'i32, 2'i32, 3'i32])
    check(m1d.dtype == Dtype.S32)
    let m2d = afa(1, 1, [1])
    check(m2d.dtype == Dtype.S32)
    let m3d = afa(2, 2, 2, @[1, 2, 3, 4, 5, 6, 7, 8])
    check(m3d.dtype == Dtype.S32)
    let m4d = afa(dim4(2, 2, 2, 2), 1..16)
    check(m4d.dtype == Dtype.S32)

test "matrix from constant":
    let m0 = constant(-1, 3, 3, Dtype.S32)
    echo m0
    echo m0.dtype
    check m0.dtype == Dtype.S32 
    check m0.len == 9
    check m0.to_seq(int) == @[-1, -1, -1, -1, -1, -1, -1, -1, -1]

    let m1 = constant(1, 2, 2, Dtype.U64)
    check(m1.dtype == DType.U64)
    check(m1.len == 4)
    check(m1.to_seq(int) == @[1, 1, 1, 1])

test "random value matrix construction":
    let m0 = randu(3, 3, Dtype.F64)
    check(m0.dtype == Dtype.F64)
    let m1 = randn(2, 2, Dtype.F32)
    check(m1.dtype == Dtype.F32)

test "matrix properties":
    let m0 = constant(10, 3, 3, Dtype.C64)
    check(m0.dtype == Dtype.C64)
    check(m0.len == 9)
    check(m0.ndims == 2)
    check(m0.dims == dim4(3, 3))
    check(m0.to_seq(int) == @[10, 10, 10, 10, 10, 10, 10, 10, 10])
    check(m0.first_as(float) == 10.0)

test "matrix indexing":
    #construct 3x3 Matrix with int32 values
    #1 4 7
    #2 5 8
    #3 6 9
    var a = afa(3, 3, 1..9, Dtype.F64)

    #first element
    let ap = a[0]
    check(a[0].first_as(int) == 1)

    #last element
    check(a[-1].first_as(int) == 9)

    #alternative last element
    check(a[iend].first_as(int) == 9)

    #second to last element
    check(a[iend-1].first_as(int) == 8)

    #second row
    check(a[1, span].to_seq(int) == @[2, 5, 8])

    #last row
    check(a.row(iend).to_seq(int) == @[3, 6, 9])

    #all but first row
    check(a.cols(1, iend).to_seq(int) == @[4, 5, 6, 7, 8, 9])

    #assign value to view spanning all elements
    a[span] = 4
    check(a.to_seq(int) == @[4, 4, 4, 4, 4, 4, 4, 4, 4])

    #set first row to 0
    a[0, span] = 0
    check(a.to_seq(int) == @[0, 4, 4, 0, 4, 4, 0, 4, 4])
