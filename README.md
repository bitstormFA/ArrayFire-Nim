# ArrayFireNim

ArrayFireNim is a [Nim](http://www.nim-lang.org) wrapper for [Arrayfire](https://github.com/arrayfire/arrayfire).

It enables *very fast* AFArray operations on different backends (CPU, OpenCL, CUDA) 

Compilation requires the C++ backend of nim (compile with cpp option)

The wrapper is using the unified backend making it is possible to switch backends at runtime.

Currently this package is developed and tested on Arch Linux.

On Windows the package is not working, any support to get it working on Windows or other platforms is highly welcome!


## Please Note
  ArrayFire-Nim is not affiliated with or endorsed by ArrayFire. The ArrayFire literal 
  mark is used under a limited license granted by ArrayFire the trademark holder 
  in the United States and other countries.

### General considerations and differences to the wrapped C++ syntax


The wrapper has been generated with [c2nim](https://github.com/nim-lang/c2nim)  but was modified to avoid name conflicts and to follow the naming conventions of Nim

The main differences from the C++ api are:

* `array` has been renamed to `AFArray` to avoid conflicts with the Nim `array` type

* `array_proxy` has been renamed to `AFArray_View`

* `seq` has been renamed to `AF_Seq` to avoid conflicts with the Nim `seq` type

* `DimT` is used for dimension types and set to clonglong on 64bit os or cint on 32bit os

* All types names are upper case

* All enums are pure except for DType and the AF_ prefix has been removed from the values

* Indexing is using square brackets instead of round brackets

* Some procs have a "m" prefix compared to the c++ functions to avoid name conflicts (e.g. msum)

* Some functions from the c++ api returning scalar values have been replaced be mulitple procs with "as_int", "as_float", "as_complex" suffixes to simplify common use cases 


The values in the documentation in the docs directory have been generated with `nim doc2` and are valid for 64bit os systems. 
For the correct values on 32bit os systems please run "nake doc".

### Known issues

The setBackend proc works but on program exit a segmentation violation will be raised.

### Current Status

Most functions from the c++ api should have been wrapped but not all have been tested.
The wrapper is already well usable but has not been optimized.

All tests have been performed on Arch Linux but with the basic libraries installed all common linux distributions should work - no tests have been performed on other OS.

### License

BSD 3-Clause License

## Types


The nim type of a AFArray is not generic - it does not depend on the type of the elements. 
The type of the AFArray elements can be checked with the `dtype` proc.

The `DType` enum contains all possible types.

To simplify cross platform application development two special values are defined.

`sysint` will be set to s32 on 32bit os systems and to s64 on 64bit os systems

`sysfloat` will be set to f32 on 32bit os systems and to f64 on 64bit os systems



## AFArray construction 


A AFArray can be constructed from an openarray, a slice, a AFArray, a sequence or a constant value.
The dimensions of the AFArray can be defined vararg of integers (max 4)
or as a Dim4 object.
If the element type of the AFArray is not defined, the nim type of the input (e.g. openarray)
will be used. On 64bit os systems literal int values will be translated to signed 64 float to float 64.

Construction of a 1,2,3,4-D AFArray from a sequence or slice without explicit type definition

```nim
    # AFArray from a sequence, AFArray type is int which maps to s64 (or s32 on 32 bit os)
    let m1d = afa(9,@[1,2,3,4,5,6,7,8,9])
    check(m1d.dtype == sysint)
    let m2d = afa(3,3,@[1,2,3,4,5,6,7,8,9])
    let m3d = afa(2,2,2,@[1,2,3,4,5,6,7,8])

    let mydims=dim4(2,2,2,2)
    let m4d = afa(mydims,1..16)                 #use a Dim4 to specify dimensions
```

Same with explicit AFArray type

```nim
    let m1d = afa(9,@[1,2,3,4,5,6,7,8,9],f64)    #float64 AFArray
    let m2d = afa(3,3,@[1,2,3,4,5,6,7,8,9],f32)  #float32 AFArray
    let m3d = afa(2,2,2,@[1,2,3,4,5,6,7,8],u64)  #usigned int 64 AFArray
    let m4d = AFArray(2,2,2,2,1..16,c64)             #complex64 AFArray
```

Construction from a constant value:

```nim
    #3x3 AFArray with all elements 0, type f64 (float64)
    let m0 = constant(0,3,3,f64)

    #2x2 AFArray with all elements 1, type taken from literal(int) -> s64 on 64bit os else s32    
    let m1 = constant(1,2,2)
```

Construction from random values:

```nim
    #3x3 AFArray with elements taken from a uniform distribution of type f64
    let m0 = randu(3,3,f64)
    #2x2 AFArray with elements taken from a normal distribution of type f32 (default)
    let m1 = randn(2,2)
```


## AFArray properties

* `len`
  Number of elements in a AFArray

* `dtype`
  Type of the AFArray elements

* `to_seq(typedesc)`
  Get all elements of a AFArray. 
  This proc takes a typedesc to define the target type, see the example below

* `first_as(typedesc)`
  Get the first element of a AFArray
  This proc takes a typedesc to define the target type, see the example below

* `dims`
  Get a Dim4 object containing the AFArray dimensions

* `ndims` 
  Get the number of dimentsions of a AFArray




```nim

    #3x3 AFArray with Complex64 elements, all set (10,0i)
    let m0 = constant(10,3,3,c64) 

    #dtype c64 
    check(m0.dtype == c64)

    #9 elements
    check(m0.len == 9)

    #2 dimensional
    check(m0.ndims == 2)

    #dim4(3,3) dimensions
    check(m0.dims == dim4(3,3))

    #all elements converted to in sequences
    check(m0.to_seq(int) == @[10,10,10,10,10,10,10,10,10] )

    #first element converted to float
    check(m0.first_as(float) == 10.0)
```

## AFArray indexing

AFArray indexing generates "views" of a AFArray based on selection criteria.
A `AFArray_View` can be assigned values and be used like a AFArray enabling very concise constructs.
The special constants `span` and `iend` are used to denote all elements / the last element 

Negative index values count backwards from the last element (i.e. iend = -1)

```nim
    #construct 3x3 AFArray with int32 values
    # 1 4 7
    # 2 5 8 
    # 3 6 9
    var a = afa(3,3, 1..9,s32)
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
```

## Backend selection

The wrapper is using the unified backend so that the backend can be changed at runtime. 
Array constructed on one backend can not be used on a different backend

`get_available_backends` returns a list of backends available. 
`setBackend` switches backend.

If a backend can access multiple devices, a device can be selected with `setDevice`

```nim

    let backends = get_available_backends()
    echo "available backends $1" % $backends
    for b in backends:
      echo "testing backend $1" % $b
      setBackend(b)
      info()
      var a = randu(3,3)
      var asum = a.sum_as_int
```


## Parallel for loops

The c++ api enables parallel `for` loops with `gfor`. 
This has been adapted to nim with the `gfor` template.

Iterations are performed in parallel by tiling the input.

```nim
    let days  = 9 
    let sites = 4
    let n     = 10

    let dayI= @[0, 0, 1, 2, 5, 5, 6, 6, 7, 8]
    let siteI = @[2, 3, 0, 1, 1, 2, 0, 1, 2, 1]
    let measurementI = @[9, 5, 6, 3, 3, 8, 2, 6, 5, 10]

    let day = afa(n,dayI)
    let site= afa(n,siteI)
    let measurement = afa(n,measurementI)

    var rainfall = constant(0,sites)    

    gfor(s, sites):
      rainfall[s] = msum(measurement * ( site == s)  )
```

## Graphics

To use the graphics functions a `window` instance is required which can be constructed with the `window` template.

```nim
    window(myWindow, 1024, 1024, "2D Vector Field example")
    # mywindow is now a var containing the window
```

## Examples

The test directory contains unit tests which have been translated from the c++ examples. 

