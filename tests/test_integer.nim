import ArrayFire_Nim
import unittest

test "basic logic and bitwise operations":
    set_backend(Backend.AF_BACKEND_DEFAULT)
    var a = afa(3,3, @[1, 2, 4, -1, 2, 0, 4, 2, 3])
    var b = afa(3,3,@[2, 3, -5, 6, 0, 10, -12, 0, 1])

    let ac0 = a.col(0) # access first column
    let ar0 = a.row(0) # access first row

    check ac0.to_seq(int) == @[1,2,4]
    check ar0.to_seq(int) == @[1,-1,4]

    a[0] = 11
    a[1] = 100

    check a[0].first_as(int) == 11  # check content of element 0
    check a[1].first_as(int) == 100  # check content of element 1

    a[1, span] = b[2, span]  # set first column from a to content of first column in b
    check a.to_seq(int) == @[11, -5, 4, -1, 10, 0, 4, 1, 3]

    let a_and_b = a & b
    check a_and_b.to_seq(int) == @[2, 3, 0, 6, 0, 0, 4, 0, 1]

    let a_or_b = a | b 
    check a_or_b.to_seq(int) == @[11, -5, -1, -1, 10, 10, -12, 1, 3]

    let a_not_b = a ^ b 
    check a_not_b.to_seq(int) == @[9, -8, -1, -7, 10, 10, -16, 1, 2] 

    let a_land_b = a && b
    check a_land_b.to_seq(int) == @[1, 1, 1, 1, 0, 0, 1, 0, 1]

    let a_lor_b = a || b 
    check a_lor_b.to_seq(int) == @[1, 1, 1, 1, 1, 1, 1, 1, 1]

    let a_transpose = a.T 
    check a_transpose.to_seq(int) == @[11, -1, 4, -5, 10, 1, 4, 0, 3]

    let a_flipv = a.flip(0)
    check a_flipv.to_seq(int) == @[4, -5, 11, 0, 10, -1, 3, 1, 4]

test "equal check arrays":
    let a = afa(3,3, 1..9)
    let a2 = afa(3,3, 1..9)
    let a3 = afa(3,3, 1..9, Dtype.F32)
    let b = constant(-1, 3,3, Dtype.S32)
    check (a == a3).alltrue(-1).to_seq(int) == @[1,1,1]




