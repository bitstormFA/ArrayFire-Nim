import unittest
import strutils
import ArrayFireNim


suite "getting started":
  test "integer":
    setDevice(0)
    info()

    echo "\n== Matrix creation"
    let h_A = @[1, 2, 4, -1, 2, 0, 4, 2, 3]
    let h_B = @[2, 3, -5, 6, 0, 10, -12, 0, 1]
    var A = matrix(3, 3, h_A);
    var B = matrix(3, 3, h_B);

    echo $A
    print("A.col(0)",A.col(0))
    print("A.row(0)",A.row(0))
    A[0] = 11
    A[1] = 100
    echo $A
    echo $B


    A[1,span] = B[2,span]
    echo $A

    echo "\n-- Bitwise operations"
    print("A & B", A&B)
    print("A | B",A|B)
    print("A ^ B",A ^ B)

    echo "\n-- Logical operations"
    print("A && B",A && B)

    echo "\n-- Transpose"
    echo $A
    print("Transpose",A.T)
    print("Hermitian Transpose",A.H)


    echo("\n-- Flip");
    echo $A
    print("Flip Dim0",A.flip(0))
    print("Flip Dim1",A.flip(1))

    echo "\n-- Sum along columns"
    echo $A
    print("A.sum", A.msum )

    echo "\n-- Product along columns"
    echo $A
    print("A.product",A.product)

    echo "\n-- Minimum along columns"
    echo $A
    print("A.mmin",A.mmin)

    echo "\n-- Maximum along columns"
    echo $A
    print("A.mmax",A.mmax)



    echo "\n-- Minimum along columns with index"
    echo $A
    var idx = Matrix()
    var mout = Matrix()
    min(mout,idx,A)
    print("OUT",mout)
    print("idx",idx)

  test "convolve":
    setDevice(0)
    info()
    let h_dx=      @[1'f32/12'f32, 8'f32/12'f32, 0'f32, 8'f32/12'f32, 1'f32/12'f32,]
    let h_spread = @[1'f32/5'f32, 1'f32/5'f32, 1'f32/5'f32, 1'f32/5'f32,]

    var img = randu(640,480)
    var dx = matrix(5,1,h_dx)
    var spread = matrix(1,5,h_spread)

    echo $dx
    echo $spread

    echo $dx.dims
    echo $spread.dims

    var kernel = matmul(dx,spread,MatProp.NONE,MatProp.NONE)


    var fulltime, dseptime : float 
    timeit(fulltime):
      var full_out = convolve2(img,kernel) 

    timeit(dseptime):
      var dsep_out = convolve(dx,spread,img)


    echo "full 2d convolution: $1" % fulltime.formatFloat(precision=4)
    echo "separable, device pointers: $1" % dseptime.formatFloat(precision=4)

  test "rainfall":
    setDevice(0)
    info()
    let days  = 9 
    let sites = 4
    let n     = 10

    let dayI= @[0, 0, 1, 2, 5, 5, 6, 6, 7, 8]
    let siteI = @[2, 3, 0, 1, 1, 2, 0, 1, 2, 1]
    let measurementI = @[9, 5, 6, 3, 3, 8, 2, 6, 5, 10]

    let day = matrix(n,dayI)
    let site= matrix(n,siteI)
    let measurement = matrix(n,measurementI)

    var rainfall = constant(0,sites, ty=DType.s32)    

    gfor(s, sites):
      rainfall[s] = msum(measurement * ( site == s)  )
    
    echo "total rainfall for each site $1\n" % $rainfall
    check(rainfall.to_seq(int) == @[8,22,22,5])

    var is_between = (1 <= day) && (day <= 5)
    var rain_between = sum_as_int(measurement * is_between)

    echo "rain between days: $1\n" % $rain_between
    check(rain_between == 20)

    var rainy_days = sum_as_int(diff1(day) > 0) + 1

    echo "number of days with rain: " & $rainy_days
    check(rainy_days == 7)

    var per_day = constant(0, days, DType.f32)

    gfor(d, days):
      per_day[d] = msum(measurement * (day == d))


    echo "total rainfall each day: $1 \n" % $per_day
    check(per_day.to_seq(int) == @[14,6,3,0,0,11,8,5,10])

    var days_over_five = sum_as_int( per_day > 5)
    echo "number of days > 5: " & $days_over_five
    check(days_over_five == 5)
