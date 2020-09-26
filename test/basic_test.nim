import unittest
import strutils
import ArrayFireNim


suite "basic test":

  test "backend test":
    let backends = get_available_backends()
    echo "available backends $1" % $backends
    for b in backends:
      echo "testing backend $1" % $b
      #setBackend(b)  #check why this results in seg violation
      #info()

  test "matrix construction without matrix type":
    let m1d = matrix(9,@[1'i32,2'i32,3'i32])
    check(m1d.dtype == s32)
    let m2d = matrix(1,1,[1])
    check(m2d.dtype == s32)
    let m3d = matrix(2,2,2,@[1,2,3,4,5,6,7,8])
    let m4d = matrix(dim4(2,2,2,2),1..16)

  test "matrix construction with specified matrix type":
    let m1d = matrix(9,@[1,2,3,4,5,6,7,8,9],f64)
    check(m1d.dtype == f64)
    let m2d = matrix(3,3,@[1,2,3,4,5,6,7,8,9],f32)
    check(m2d.dtype == f32)
    let m3d = matrix(2,2,2,@[1,2,3,4,5,6,7,8],u64)
    check(m3d.dtype == u64)
    let m4d = matrix(2,2,2,2,1..16,c64)
    check(m4d.dtype == c64)

  test "matrix from constant":
    let m0 = constant(0,3,3,f64)
    check(m0.dtype == f64)
    check(m0.len == 9)
    check(m0.to_seq(int)  == @[0,0,0,0,0,0,0,0,0] )

    let m1 = constant(1,2,2, u64)
    check(m1.dtype == DType.u64)
    check(m1.len == 4)
    check(m1.to_seq(int)  == @[1,1,1,1] )

  test "random value matrix construction":
    let m0 = randu(3,3,f64)
    check(m0.dtype == f64)
    let m1 = randn(2,2)
    check(m1.dtype == f32)

  test "matrix properties":
    let m0 = constant(10,3,3,c64)
    check(m0.dtype == c64)
    check(m0.len == 9)
    check(m0.ndims == 2)
    check(m0.dims == dim4(3,3))
    check(m0.to_seq(int) == @[10,10,10,10,10,10,10,10,10] )
    check(m0.first_as(float) == 10.0)

  test "matrix indexing":
    #construct 3x3 Matrix with int32 values
    # 1 4 7
    # 2 5 8 
    # 3 6 9
    var a = matrix(3,3, 1..9,s32)

    #first element
    check(a[0].first_as(int) == 1 )

    #last element
    check(a[-1].first_as(int) == 9 )

    #also last element
    check(a[iend].first_as(int) == 9 )

    #second to last element
    check(a[iend-1].first_as(int) == 8 )

    #second row
    check(a[1,span].to_seq(int) == @[2,5,8])

    #last row
    check(a.row(iend).to_seq(int) == @[3,6,9])

    #all but first row
    check(a.cols(1,iend).to_seq(int) == @[4,5,6,7,8,9] )

    #assign value to view spanning all elements
    a[span] = 4
    check(a.to_seq(int) == @[4,4,4,4,4,4,4,4,4])

    #set first row to 0
    a[0,span] = 0
    check(a.to_seq(int) == @[0,4,4,0,4,4,0,4,4])
