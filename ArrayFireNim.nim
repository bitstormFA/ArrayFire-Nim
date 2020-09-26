import strutils
import tables
import times
import typetraits

when defined(Windows): #check this
  {.passC: "-std=c++11".}
  {.passL: "-lGL -laf"}
elif defined(Linux):
  {.passC: "-std=c++11".}
  {.passL: "-lGL -laf"}
elif defined(MacOsX):
  {.passC: "-std=c++11".}
  {.passL: "-laf"}

when sizeof(int) == 8:
  type DimT* = clonglong
  const dimt_cdef = """
  typedef long long DimT;

  """
else:
  type DimT* = cint
  const dimt_cdef = """
  typedef int DimT;

  """    

{.emit: dimt_cdef.}

##[
ArrayFireNim
============

ArrayFireNim is a `Nim<http://www.nim-lang.org>`_ wrapper for `Arrayfire<https://github.com/arrayfire/arrayfire>`_.

It enables very fast matrix operations on different backends (CPU, OpenCL, CUDA) 

Compilation requires the C++ backend of Nim (compile with cpp option)

The wrapper is using the unified backend making it is possible to switch backends at runtime.

General considerations and differences to the wrapped C++ syntax
-----------------------------------------------------------------

The wrapper has been generated with `c2nim<https://github.com/nim-lang/c2nim>`_ but was 
modified to avoid name conflicts and to follow the naming conventions of Nim

The main differences from the C++ api are:
* ``array`` has been renamed to ``Matrix`` to avoid conflicts with the Nim ``array`` type
* ``array_proxy`` has been renamed to ``Matrix_View``
* ``seq`` has been renamed to AF_Seq to avoid conflicts with the nim ``seq`` type
* ``DimT`` is used for dimension types and set to clonglong on 64bit os or cint on 32bit os
* All types names are upper case
* All enums are pure except for DType and the AF_ prefix has been removed from the values
* Indexing is using square brackets instead of round brackets
* Some procs have a "m" prefix compared to the c++ functions to avoid name conflicts (e.g. msum)
* Some functions from the c++ api returning skalar values have been replaced be mulitple procs with "as_int", "as_float", "as_complex" suffixes to simplify common use cases

The values in the documentation in docs have been generated with ``nim doc2`` are valid for 64bit os systems. 
For the correct values on 32bit os systems please run "nake doc".

Types
-----------

The Nim type of a Matrix is not generic - it does not depend on the type of the elements. 
The type of the matrix elements can be checked with the ``dtype`` proc.

The ``DType`` enum contains all possible types.

To simplify cross platform application development two special values are defined.

``sysint`` will be set to s32 on 32bit os systems and to s64 on 64bit os systems

``sysfloat`` will be set to f32 on 32bit os systems and to f64 on 64bit os systems



Matrix construction 
--------------------

A Matrix can be constructed from an openarray, a slice, a matrix, a sequence or a constant value.
The dimensions of the matrix can be defined vararg of integers (max 4)
or as a Dim4 object.
If the element type of the matrix is not defined, the Nim type of the input (e.g. openarray)
will be used. On 64bit os systems literal int values will be translated to signed 64 float to float 64.

Construction of a 1,2,3,4-D matrix from a sequence or slice without explicit type definition

.. code-block:: nim
    # Matrix from a sequence, matrix type is int which maps to s64 (or s32 on 32 bit os)
    let m1d = matrix(9,@[1,2,3,4,5,6,7,8,9])
    check(m1d.dtype == sysint)
    let m2d = matrix(3,3,@[1,2,3,4,5,6,7,8,9])
    let m3d = matrix(2,2,2,@[1,2,3,4,5,6,7,8])

    let mydims=dim4(2,2,2,2)
    let m4d = matrix(mydims,1..16)                 #use a Dim4 to specify dimensions


Same with explicit matrix type

.. code-block:: nim
    let m1d = matrix(9,@[1,2,3,4,5,6,7,8,9],f64)    #float64 matrix
    let m2d = matrix(3,3,@[1,2,3,4,5,6,7,8,9],f32)  #float32 matrix
    let m3d = matrix(2,2,2,@[1,2,3,4,5,6,7,8],u64)  #usigned int 64 matrix
    let m4d = matrix(2,2,2,2,1..16,c64)             #complex64 matrix


Construction from a constant value:

.. code-block:: nim
    #3x3 Matrix with all elements 0, type f64 (float64)
    let m0 = constant(0,3,3,f64)

    #2x2 Matrix with all elements 1, type taken from literal(int) -> s64 on 64bit os else s32    
    let m1 = constant(1,2,2)


Construction from random values:

.. code-block:: nim
    #3x3 Matrix with elements taken from a uniform distribution of type f64
    let m0 = randu(3,3,f64)
    #2x2 Matrix with elements taken from a normal distribution of type f32 (default)
    let m1 = randn(2,2)



Matrix properties
-----------------

``len``
  Number of elements in a matrix

``dtype``
  Type of the matrix elements

``to_seq(typedesc)``
  Get all elements of a matrix. 
  This proc takes a typedesc to define the target type, see the example below

``first_as(typedesc)``
  Get the first element of a matrix
  This proc takes a typedesc to define the target type, see the example below

``dims``
  Get a Dim4 object containing the matrix dimensions

``ndims`` 
  Get the number of dimentsions of a matrix




.. code-block:: nim

    #3x3 Matrix with Complex64 elements, all set (10,0i)
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


Matrix indexing
---------------

Matrix indexing generates "views" of a matrix based on selection criteria.
A ``Matrix_View`` can be assigned values and be used like a matrix enabling
very concise constructs.
The special constants ``span`` and ``iend`` are used to denote all elements / the last element
Negative index values count backwards from the last element (i.e. iend = -1)
See the ``[]`` procs for all details.

.. code-block:: nim
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


Backend selection
-----------------

The wrapper is using the unified backend so that the backend can be changed at runtime
Array constructed on one backend can not be used on a different backend

``get_available_backends`` returns a list of backends available. 
``setBackend`` switches backend.

If a backend can access multiple devices, a device can be selected with ``setDevice``

.. code-block:: nim

    let backends = get_available_backends()
    echo "available backends $1" % $backends
    for b in backends:
      echo "testing backend $1" % $b
      setBackend(b)
      info()
      var a = randu(3,3)
      var asum = a.sum_as_int



Parallel for loops
------------------

The c++ api enables parallel ``for`` loops with ``gfor``. 
This has been adapted to Nim with the ``gfor`` template.

Iterations are performed in parallel by tiling the input.

.. code-block:: nim
    let days  = 9 
    let sites = 4
    let n     = 10

    let dayI= @[0, 0, 1, 2, 5, 5, 6, 6, 7, 8]
    let siteI = @[2, 3, 0, 1, 1, 2, 0, 1, 2, 1]
    let measurementI = @[9, 5, 6, 3, 3, 8, 2, 6, 5, 10]

    let day = matrix(n,dayI)
    let site= matrix(n,siteI)
    let measurement = matrix(n,measurementI)

    var rainfall = constant(0,sites)    

    gfor(s, sites):
      rainfall[s] = msum(measurement * ( site == s)  )



Graphics
--------

To use the graphics functions a ``window`` which can be constructed with ``window`` template.

.. code-block:: nim
    window(myWindow, 1024, 1024, "2D Vector Field example")
    # mywindow is now a var containing the window


Examples
--------
The test directory contains unit tests which have been translated from the c++ examples. 


*Please Note:*
--------------
  ArrayFire-Nim is not affiliated with or endorsed by ArrayFire. The ArrayFire literal 
  mark is used under a limited license granted by ArrayFire the trademark holder 
  in the United States and other countries.
]##

type
  intl* = clonglong
  uintl* = culonglong
  Err* {.pure, size: sizeof(cint).} = enum
    SUCCESS = 0, ERR_NO_MEM = 101, ERR_DRIVER = 102, ERR_RUNTIME = 103,
    ERR_INVALID_ARRAY = 201, ERR_ARG = 202, ERR_SIZE = 203, ERR_TYPE = 204,
    ERR_DIFF_TYPE = 205, ERR_BATCH = 207, ERR_DEVICE = 208, ERR_NOT_SUPPORTED = 301,
    ERR_NOT_CONFIGURED = 302, ERR_NONFREE = 303, ERR_NO_DBL = 401, ERR_NO_GFX = 402,
    ERR_LOAD_LIB = 501, ERR_LOAD_SYM = 502, ERR_ARR_BKND_MISMATCH = 503,
    ERR_INTERNAL = 998, ERR_UNKNOWN = 999
  
  Dtype* {.size: sizeof(cint), header : "arrayfire.h", importcpp: "af_dtype".} = enum ## \
    ##
    ## ====== ========
    ## DType  Nim type
    ## ====== ========
    ## f32    float32
    ## c32    Complex32
    ## f64    float64
    ## c64    Complex64
    ## b8     bool
    ## s32    int32
    ## u32    uint32
    ## u8     uint8
    ## s64    int64
    ## u64    uint64
    ## s16    int16
    ## u16    uint16
    ## ====== ========
    f32, c32, f64, c64, b8, s32, u32, u8, s64, u64, s16, u16 

  Source* {.pure, size: sizeof(cint), header : "arrayfire.h", importcpp: "af_source".} = enum
    afDevice, afHost

type
  InterpType* {.pure, size: sizeof(cint).} = enum
    NEAREST, LINEAR, BILINEAR, CUBIC, LOWER,
    LINEAR_COSINE, BILINEAR_COSINE, BICUBIC,
    CUBIC_SPLINE, BICUBIC_SPLINE

  BorderType* {.pure, size: sizeof(cint).} = enum
    PAD_ZERO = 0, PAD_SYM, PAD_CLAMP_TO_EDGE, PAD_PERIODIC

  Connectivity* {.pure, size: sizeof(cint).} = enum
    CONNECTIVITY_4 = 4, CONNECTIVITY_8 = 8

  ConvMode* {.pure, size: sizeof(cint).} = enum
    DEFAULT, EXPAND

  ConvDomain* {.pure, size: sizeof(cint).} = enum
    AUTO, SPATIAL, FREQ

  ConvGradientType* {.pure, size: sizeof(cint).} = enum
    DEFAULT, FILTER, DATA, BIAS

  FluxFuction* {.pure, size: sizeof(cint).} = enum
    DEFAULT = 0, QUADRATIC = 1, EXPONENTIAL = 2 

  CannyThreshold* {.pure, size: sizeof(cint).} = enum
    THRESHOLD_MANUAL = 0, THRESHOLD_AUTO_OTSU = 1 

  DiffusionEq* {.pure, size: sizeof(cint).} = enum
    DEFAULT = 0, GRAD = 1, MCDE = 2 

  InverseDeconvAlgo* {.pure, size: sizeof(cint).} = enum
    DEFAULT = 0, IKHONOV = 1

  MatchType* {.pure, size: sizeof(cint).} = enum
    SAD = 0, ZSAD, LSAD, SSD, ZSSD, LSSD, NCC, ZNCC, SHD

  YccStd* {.pure, size: sizeof(cint).} = enum
    YCC_601 = 601, YCC_709 = 709, YCC_2020 = 2020

  CspaceT* {.pure, size: sizeof(cint).} = enum
    GRAY = 0, RGB, HSV, YCbCr

  MatProp* {.pure, size: sizeof(cint), header : "arrayfire.h", importcpp: "af_matprop".} = enum
    NONE = 0, TRANS = 1, CTRANS = 2, CONJ = 4, UPPER = 32, LOWER = 64,
    DIAG_UNIT = 128, SYM = 512, POSDEF = 1024, ORTHOG = 2048,
    TRI_DIAG = 4096, BLOCK_DIAG = 8192

  NormType* {.pure, size: sizeof(cint).} = enum
    VECTOR_1, VECTOR_INF, VECTOR_2, VECTOR_P, MATRIX_1,
    MATRIX_INF, MATRIX_2, MATRIX_L_PQ

  ImageFormat* {.pure, size: sizeof(cint).} = enum
    BMP = 0, ICO = 1, JPEG = 2, JNG = 3, PNG = 13, PPM = 14,
    PPMRAW = 15, TIFF = 18, PSD = 20, HDR = 26, EXR = 29, JP2 = 31,
    RAW = 34

  MomentType* {.pure, size: sizeof(cint).} = enum
    M00 = 1, M01 = 2, M10 = 4, M11 = 8, FIRST_ORDER = 15

  HomographyType* {.pure, size: sizeof(cint).} = enum
    RANSAC = 0, LMEDS = 1

  TopKFunction* {.pure, size: sizeof(cint).} = enum
    TOPK_DEFAULT = 0, TOPK_MIN = 1, TOPK_MAX = 2,

type
  Backend* {.size: sizeof(cint), header : "arrayfire.h", importcpp: "af_backend".} = enum
    UNIFIED = 0, CPU = 1, CUDA = 2, OPENCL = 4

type
  SomeenumT* {.pure, size: sizeof(cint).} = enum
    ID = 0

  BinaryOp* {.pure, size: sizeof(cint).} = enum
    BINARY_ADD = 0, BINARY_MUL = 1, BINARY_MIN = 2, BINARY_MAX = 3

  RandomEngineType* {.pure, size: sizeof(cint), header : "arrayfire.h", 
    importcpp: "af_random_engine_type".} = enum
    PHILOX = 100, THREEFRY = 200,
    MERSENNE = 300,

type
  Colormap* {.pure, size: sizeof(cint).} = enum
    DEFAULT = 0, SPECTRUM = 1, COLORS = 2, RED = 3,
    MOOD = 4, HEAT = 5, BLUE = 6, INFERNO = 7, MAGMA = 8, PLASMA = 9, VIRIDS = 10

  MarkerType* {.pure, size: sizeof(cint).} = enum
    NONE = 0, POINT = 1, CIRCLE = 2, SQUARE = 3,
    TRIANGLE = 4, CROSS = 5, PLUS = 6, STAR = 7

  Storage* {.pure, size: sizeof(cint).} = enum
    DENSE = 0, CSR = 1, CSC = 2, COO = 3


type
  CSpace* = CspaceT
  SomeEnum* = SomeenumT


type
  Trans* = MatProp

type
  Matrix* {.final, header : "arrayfire.h", importcpp: "af::array".} = object
  Matrix_View* {.final, header : "arrayfire.h", importcpp: "af::array::array_proxy".} = object
  Dim4* {.final, header : "arrayfire.h", importcpp: "af::dim4".} = object
  RandomEngine* {.final, header : "arrayfire.h", importcpp: "af::randomEngine".} = object
  AF_Seq* {.final, header : "arrayfire.h", importcpp: "af::seq".} = object
  Index* {.final, header : "arrayfire.h", importcpp: "af::index".} = object
  Window* {.final, header : "arrayfire.h", importcpp: "af::Window", shallow.} = object
  Timer* {.final, header : "arrayfire.h", importcpp: "af::timer".} = object
  AFC_RandomEngine = object
  Features* {.final, header : "arrayfire.h", importcpp: "af::features".} = object
  AFC_Features* = object

type 
  AF_Array_Handle* = distinct pointer

  AFC_Seq* = object
    begin*: cdouble
    until*: cdouble
    step*: cdouble

  IndexOption* {.union.} = object 
    arr*: Matrix
    aseq*: AFC_Seq

  IndexT* = object
    idx*: pointer
    isSeq*: bool
    isBatch*: bool


type
  Complex32* {.final, header : "arrayfire.h", importcpp: "af::cfloat".} = object
    real*: cfloat
    imag*: cfloat

  Complex64* {.final, header : "arrayfire.h", importcpp: "af::cdouble".} = object
    real*: cdouble
    imag*: cdouble


type
  BatchFuncT* = proc (lhs: Matrix; rhs: Matrix): Matrix {.cdecl.}

type
  Cell* = object
    row*: cint
    col*: cint
    title*: cstring
    cmap*: Colormap

when sizeof(int)==8:
   const sysint* = DType.s64
   const sysuint* = DType.u64
else:
   const sysint* = DType.s32
   const sysuint* = DType.u32

when sizeof(float)==8:
   const sysfloat* = DType.f64
else:
   const sysfloat=DType.f32



proc constructdim4*(first: DimT; second: DimT = 1; third: DimT = 1; fourth: DimT = 1): Dim4 
  {.cdecl, constructor, importcpp: "af::dim4(@)", header : "arrayfire.h".}

proc constructdim4*(other: Dim4): Dim4 
  {.cdecl, constructor, importcpp: "af::dim4(@)",header : "arrayfire.h".}

proc constructdim4*(ndims: cuint; dims: ptr DimT): Dim4 
  {.cdecl, constructor, importcpp: "dim4(@)", header : "arrayfire.h".}

proc elements*(this: var Dim4): DimT 
  {.cdecl, importcpp: "elements", header : "arrayfire.h".}

proc elements*(this: Dim4): DimT 
  {.noSideEffect, cdecl, importcpp: "elements", header : "arrayfire.h".}

proc ndims*(this: var Dim4): DimT 
  {.cdecl, importcpp: "ndims", header : "arrayfire.h".}

proc ndims*(this: Dim4): DimT 
  {.noSideEffect, cdecl, importcpp: "ndims", header : "arrayfire.h".}

proc `==`*(this: Dim4; other: Dim4): bool 
  {.noSideEffect, cdecl, importcpp: "(# == #)",header : "arrayfire.h".}

proc `*=`*(this: var Dim4; other: Dim4) 
  {.cdecl, importcpp: "(# *= #)", header : "arrayfire.h".}

proc `+=`*(this: var Dim4; other: Dim4) 
  {.cdecl, importcpp: "(# += #)", header : "arrayfire.h".}

proc `-=`*(this: var Dim4; other: Dim4) 
  {.cdecl, importcpp: "(# -= #)", header : "arrayfire.h".}

proc `[]`*(this: var Dim4; dim: cuint): var DimT 
  {.cdecl, importcpp: "#[@]", header : "arrayfire.h".}

proc `[]`*(this: Dim4; dim: cuint): DimT 
  {.noSideEffect, cdecl, importcpp: "#[@]",header : "arrayfire.h".}

proc get*(this: var Dim4): ptr DimT 
  {.cdecl, importcpp: "get", header : "arrayfire.h".}

proc get*(this: Dim4): ptr DimT 
  {.noSideEffect, cdecl, importcpp: "get", header : "arrayfire.h".}

proc `+`*(first: Dim4; second: Dim4): Dim4 
  {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}

proc `-`*(first: Dim4; second: Dim4): Dim4 
  {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}

proc `*`*(first: Dim4; second: Dim4): Dim4 
  {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}

proc dim4s*[T: int | DimT](dims : openarray[T]) : Dim4 =
  var all_dims = [DimT(1),DimT(1),DimT(1),DimT(1)]
  let count = min(4,len(dims))  
  for i in 0..<count:
    all_dims[i]=Dimt(dims[i])
  constructdim4(all_dims[0], all_dims[1], all_dims[2], all_dims[3])

proc dim4*[T: int | DimT](dims: varargs[T]) : Dim4 =
  dim4s(dims)

converter toDimT*(i: int) : DimT = DimT(i) ##automatically convert a single int to a DimTS

proc `[]`*(d : Dim4, i:int) : int =
  ##Index access to the dim4 dimensions
  int(d[cuint(i)])


proc `$`*(d : Dim4) : string =
  var elems=newSeq[DimT]()
  for i in 0..<4:
    elems.add(DimT(d[i]))  
  "Dim4[$1]"%join(elems,", ")  


proc `$`*(c: Complex32) : string ="CF($1 + $2i)" % [$c.real,$c.imag]
proc `$`*(c: Complex64) : string ="CD($1 + $2i)" % [$c.real,$c.imag]


proc complex32*[R,I](r:R, i:I) : Complex32 = Complex32(real:cfloat(r),imag:cfloat(i)) 
proc complex64*[R,I](r:R, i:I) : Complex64 = Complex64(real:cdouble(r),imag:cdouble(i)) 

proc complex32*[R](r:R) : Complex32 = 
  when R is Complex32:
    r
  elif R is Complex64:
    Complex32(r.real,r.imag)
  else:
    Complex32(real:cfloat(r), imag:cfloat(0)) 

proc complex64*[R](r:R) : Complex64 = 
  when R is Complex64:
    r
  elif R is Complex32:
    Complex64(r.real,r.imag)
  else:
    Complex64(real:cdouble(r),imag:cdouble(0)) 



converter toFloat*[T: Complex32 | Complex64](c : T) : float = float(c.real)
converter toFloat32*[T: Complex32 | Complex64](c : T) : float32 = float32(c.real)
converter toFloat64*[T: Complex32 | Complex64](c : T) : float64 = float64(c.real)
converter toBool*[T: Complex32 | Complex64](c: T) : bool = bool(c.real == 0.0 and c.imag == 0.0)
converter toInt*[T: Complex32 | Complex64](c : T) : int = int(c.real)
converter toInt16*[T: Complex32 | Complex64](c : T) : int16 = int16(c.real)
converter toInt32*[T: Complex32 | Complex64](c : T) : int32 = int32(c.real)
converter toInt64*[T: Complex32 | Complex64](c : T) : int64 = int64(c.real)
converter toUInt*[T: Complex32 | Complex64](c : T) : uint = uint(c.real)
converter toUInt8*[T: Complex32 | Complex64](c : T) : uint8 = uint8(c.real)
converter toUInt16*[T: Complex32 | Complex64](c : T) : uint16 = uint16(c.real)
converter toUInt32*[T: Complex32 | Complex64](c : T) : uint32 = uint32(c.real)
converter toUInt64*[T: Complex32 | Complex64](c : T) : uint64 = uint64(c.real)

proc set*(this: var Matrix; tmp: AF_Array_Handle) 
  {.cdecl, importcpp: "af::set", header : "arrayfire.h".}

proc matrix*(): Matrix 
  {.cdecl, constructor, importcpp: "af::array(@)",header : "arrayfire.h".}

proc matrix*(handle: AF_Array_Handle): Matrix 
  {.cdecl, constructor,importcpp: "af::array(@)", header : "arrayfire.h".}

proc matrix*(matin : Matrix): Matrix 
  {.cdecl, constructor,importcpp: "af::array(@)", header : "arrayfire.h".}

proc matrix*(dim0: DimT; ty: Dtype = f32): Matrix 
  {.cdecl, constructor, importcpp: "af::array(@)", header : "arrayfire.h".}

proc matrix*(dim0: DimT; dim1: DimT; ty: Dtype = f32): Matrix 
  {.cdecl, constructor,importcpp: "af::array(@)", header : "arrayfire.h".}

proc matrix*(dim0: DimT; dim1: DimT; dim2: DimT; ty: Dtype = f32): Matrix 
  {.cdecl,constructor, importcpp: "af::array(@)", header : "arrayfire.h".}

proc matrix*(dim0: DimT; dim1: DimT; dim2: DimT; dim3: DimT; ty: Dtype = f32): Matrix 
  {.cdecl, constructor, importcpp: "af::array(@)", header : "arrayfire.h".}

proc matrix*(dims: Dim4; ty: Dtype = f32): Matrix 
  {.cdecl, constructor,importcpp: "af::array(@)", header : "arrayfire.h".}

proc matrix*[T](dim0: DimT; pointer: ptr T; src: Source = Source.afHost): Matrix 
  {.cdecl,constructor, importcpp: "af::array(@)", header : "arrayfire.h".}

proc matrix*[T](dim0: DimT; dim1: DimT; pointer: ptr T; src: Source = Source.afHost): Matrix 
  {.cdecl, constructor, importcpp: "af::array(@)", header : "arrayfire.h".}

proc matrix*[T](dim0: DimT; dim1: DimT; dim2: DimT; pointer: ptr T;
                src: Source = Source.afHost): Matrix 
  {.cdecl, constructor,importcpp: "af::array(@)", header : "arrayfire.h".}

proc matrix*[T](dim0: DimT; dim1: DimT; dim2: DimT; dim3: DimT; pointer: ptr T;
                src: Source = Source.afHost): Matrix 
  {.cdecl, constructor, importcpp: "af::array(@)", header : "arrayfire.h".}

proc matrix*[T](dims: Dim4; pointer: ptr T; src: Source = Source.afHost): Matrix 
  {.cdecl, constructor, importcpp: "af::array(@)", header : "arrayfire.h".}

proc matrix*(input: Matrix; dims: Dim4): Matrix 
  {.cdecl, constructor, importcpp: "af::array(@)", header : "arrayfire.h".}

proc matrix*(input: Matrix; dim0: DimT; dim1: DimT = 1; dim2: DimT = 1; dim3: DimT = 1): Matrix 
  {.cdecl, constructor, importcpp: "af::array(@)", header : "arrayfire.h".}

proc get*(this: var Matrix): AF_Array_Handle 
  {.cdecl, importcpp: "get", header : "arrayfire.h".}

proc get*(this: Matrix): AF_Array_Handle 
  {.noSideEffect, cdecl, importcpp: "get", header : "arrayfire.h".}

proc elements*(this: Matrix): DimT 
  {.noSideEffect, cdecl, importcpp: "elements",header : "arrayfire.h".}

proc host*[T](this: Matrix): ptr T 
  {.noSideEffect, cdecl, importcpp: "host", header : "arrayfire.h".}

proc host*(this: Matrix; `ptr`: pointer) 
  {.noSideEffect, cdecl, importcpp: "host", header : "arrayfire.h".}

proc write*[T](this: var Matrix; `ptr`: ptr T; bytes: csize_t; src: Source = Source.afHost) 
  {.cdecl,importcpp: "write", header : "arrayfire.h".}

proc `type`*(this: Matrix): Dtype 
  {.noSideEffect, cdecl, importcpp: "type", header : "arrayfire.h".}

proc dtype*(this: Matrix): Dtype 
  {.noSideEffect, cdecl, importcpp: "type",header : "arrayfire.h".}

proc af_dims*(this: Matrix; dim: cuint): DimT 
  {.noSideEffect, cdecl, importcpp: "dims",header : "arrayfire.h".}

proc numdims*(this: Matrix): cuint 
  {.noSideEffect, cdecl, importcpp: "numdims",header : "arrayfire.h".}

proc bytes*(this: Matrix): csize_t 
  {.noSideEffect, cdecl, importcpp: "bytes", header : "arrayfire.h".}

proc copy*(this: Matrix): Matrix 
  {.noSideEffect, cdecl, importcpp: "copy", header : "arrayfire.h".}

proc isempty*(this: Matrix): bool 
  {.noSideEffect, cdecl, importcpp: "isempty",header : "arrayfire.h".}

proc isscalar*(this: Matrix): bool 
  {.noSideEffect, cdecl, importcpp: "isscalar",header : "arrayfire.h".}

proc isvector*(this: Matrix): bool 
  {.noSideEffect, cdecl, importcpp: "isvector",header : "arrayfire.h".}

proc isrow*(this: Matrix): bool 
  {.noSideEffect, cdecl, importcpp: "isrow", header : "arrayfire.h".}

proc iscolumn*(this: Matrix): bool 
  {.noSideEffect, cdecl, importcpp: "iscolumn",header : "arrayfire.h".}

proc iscomplex*(this: Matrix): bool 
  {.noSideEffect, cdecl, importcpp: "iscomplex",header : "arrayfire.h".}

proc isreal*(this: Matrix): bool 
  {.noSideEffect, cdecl, importcpp: "isreal",header : "arrayfire.h".}

proc isdouble*(this: Matrix): bool 
  {.noSideEffect, cdecl, importcpp: "isdouble",header : "arrayfire.h".}

proc issingle*(this: Matrix): bool 
  {.noSideEffect, cdecl, importcpp: "issingle",header : "arrayfire.h".}

proc isrealfloating*(this: Matrix): bool 
  {.noSideEffect, cdecl,importcpp: "isrealfloating", header : "arrayfire.h".}

proc isfloating*(this: Matrix): bool 
  {.noSideEffect, cdecl, importcpp: "isfloating",header : "arrayfire.h".}

proc isinteger*(this: Matrix): bool 
  {.noSideEffect, cdecl, importcpp: "isinteger",header : "arrayfire.h".}

proc isbool*(this: Matrix): bool 
  {.noSideEffect, cdecl, importcpp: "isbool",header : "arrayfire.h".}

proc eval*(this: Matrix) 
  {.noSideEffect, cdecl, importcpp: "eval", header : "arrayfire.h".}

proc scalar*[T](this: Matrix): T 
  {.noSideEffect, cdecl, importcpp: "#.scalar<'*0>()", header : "arrayfire.h".}

proc scalar_r*(this: Matrix): cdouble 
  {.noSideEffect, cdecl, importcpp: "#.scalar<double>()", header : "arrayfire.h".}

proc device*[T](this: Matrix): ptr T 
  {.noSideEffect, cdecl, importcpp: "device", header : "arrayfire.h".}

proc row*(this: var Matrix; index: cint): Matrix_View 
  {.cdecl, importcpp: "row", header : "arrayfire.h".}

proc row*(this: Matrix; index: cint): Matrix_View 
  {.noSideEffect, cdecl, importcpp: "row", header : "arrayfire.h".}

proc rows*(this: var Matrix; first: cint; last: cint): Matrix_View 
  {.cdecl, importcpp: "rows", header : "arrayfire.h".}

proc rows*(this: Matrix; first: cint; last: cint): Matrix_View 
  {.noSideEffect, cdecl, importcpp: "rows", header : "arrayfire.h".}

proc col*(this: var Matrix; index: cint): 
  Matrix_View {.cdecl, importcpp: "col", header : "arrayfire.h".}

proc col*(this: Matrix; index: cint): Matrix_View 
  {.noSideEffect, cdecl, importcpp: "col", header : "arrayfire.h".}

proc cols*(this: var Matrix; first: cint; last: cint): Matrix_View 
  {.cdecl, importcpp: "cols", header : "arrayfire.h".}

proc cols*(this: Matrix; first: cint; last: cint): Matrix_View 
  {.noSideEffect, cdecl, importcpp: "cols", header : "arrayfire.h".}

proc slice*(this: var Matrix; index: cint): Matrix_View 
  {.cdecl, importcpp: "slice", header : "arrayfire.h".}

proc slice*(this: Matrix; index: cint): Matrix_View 
  {.noSideEffect, cdecl, importcpp: "slice", header : "arrayfire.h".}

proc slices*(this: var Matrix; first: cint; last: cint): Matrix_View 
  {.cdecl, importcpp: "slices", header : "arrayfire.h".}

proc slices*(this: Matrix; first: cint; last: cint): Matrix_View 
  {.noSideEffect, cdecl, importcpp: "slices", header : "arrayfire.h".}

proc `as`*(this: Matrix; `type`: Dtype): Matrix 
  {.noSideEffect, cdecl, importcpp: "as", header : "arrayfire.h".}

proc destroy*(this: var Matrix) 
  {.cdecl, importcpp: "#.~array()", header : "arrayfire.h".}

proc T*(this: Matrix): Matrix 
  {.noSideEffect, cdecl, importcpp: "T", header : "arrayfire.h".}

proc H*(this: Matrix): Matrix 
  {.noSideEffect, cdecl, importcpp: "H", header : "arrayfire.h".}

proc `+=`*(this: var Matrix; val: Matrix) 
  {.cdecl, importcpp: "(# += #)", header : "arrayfire.h".}

proc `+=`*(this: var Matrix; val: cdouble) 
  {.cdecl, importcpp: "(# += #)", header : "arrayfire.h".}

proc `+=`*(this: var Matrix; val: cfloat) 
  {.cdecl, importcpp: "(# += #)", header : "arrayfire.h".}

proc `+=`*(this: var Matrix; val: cint) 
  {.cdecl, importcpp: "(# += #)", header : "arrayfire.h".}

proc `+=`*(this: var Matrix; val: cuint) 
  {.cdecl, importcpp: "(# += #)", header : "arrayfire.h".}

proc `+=`*(this: var Matrix; val: bool) 
  {.cdecl, importcpp: "(# += #)", header : "arrayfire.h".}

proc `+=`*(this: var Matrix; val: char) 
  {.cdecl, importcpp: "(# += #)", header : "arrayfire.h".}

proc `+=`*(this: var Matrix; val: clong) 
  {.cdecl, importcpp: "(# += #)", header : "arrayfire.h".}

proc `+=`*(this: var Matrix; val: culong) 
  {.cdecl, importcpp: "(# += #)", header : "arrayfire.h".}

proc `+=`*(this: var Matrix; val: clonglong) 
  {.cdecl, importcpp: "(# += #)", header : "arrayfire.h".}

proc `+=`*(this: var Matrix; val: culonglong) 
  {.cdecl, importcpp: "(# += #)", header : "arrayfire.h".}

proc `-=`*(this: var Matrix; val: Matrix) {.cdecl, importcpp: "(# -= #)", header : "arrayfire.h".}
proc `-=`*(this: var Matrix; val: cdouble) {.cdecl, importcpp: "(# -= #)", header : "arrayfire.h".}
proc `-=`*(this: var Matrix; val: cfloat) {.cdecl, importcpp: "(# -= #)", header : "arrayfire.h".}
proc `-=`*(this: var Matrix; val: cint) {.cdecl, importcpp: "(# -= #)", header : "arrayfire.h".}
proc `-=`*(this: var Matrix; val: cuint) {.cdecl, importcpp: "(# -= #)", header : "arrayfire.h".}
proc `-=`*(this: var Matrix; val: bool) {.cdecl, importcpp: "(# -= #)", header : "arrayfire.h".}
proc `-=`*(this: var Matrix; val: char) {.cdecl, importcpp: "(# -= #)", header : "arrayfire.h".}
proc `-=`*(this: var Matrix; val: clong) {.cdecl, importcpp: "(# -= #)", header : "arrayfire.h".}
proc `-=`*(this: var Matrix; val: culong) {.cdecl, importcpp: "(# -= #)", header : "arrayfire.h".}
proc `-=`*(this: var Matrix; val: clonglong) {.cdecl, importcpp: "(# -= #)",header : "arrayfire.h".}
proc `-=`*(this: var Matrix; val: culonglong) {.cdecl, importcpp: "(# -= #)",header : "arrayfire.h".}
proc `*=`*(this: var Matrix; val: Matrix) {.cdecl, importcpp: "(# *= #)", header : "arrayfire.h".}
proc `*=`*(this: var Matrix; val: cdouble) {.cdecl, importcpp: "(# *= #)", header : "arrayfire.h".}
proc `*=`*(this: var Matrix; val: cfloat) {.cdecl, importcpp: "(# *= #)", header : "arrayfire.h".}
proc `*=`*(this: var Matrix; val: cint) {.cdecl, importcpp: "(# *= #)", header : "arrayfire.h".}
proc `*=`*(this: var Matrix; val: cuint) {.cdecl, importcpp: "(# *= #)", header : "arrayfire.h".}
proc `*=`*(this: var Matrix; val: bool) {.cdecl, importcpp: "(# *= #)", header : "arrayfire.h".}
proc `*=`*(this: var Matrix; val: char) {.cdecl, importcpp: "(# *= #)", header : "arrayfire.h".}
proc `*=`*(this: var Matrix; val: clong) {.cdecl, importcpp: "(# *= #)", header : "arrayfire.h".}
proc `*=`*(this: var Matrix; val: culong) {.cdecl, importcpp: "(# *= #)", header : "arrayfire.h".}
proc `*=`*(this: var Matrix; val: clonglong) {.cdecl, importcpp: "(# *= #)",header : "arrayfire.h".}
proc `*=`*(this: var Matrix; val: culonglong) {.cdecl, importcpp: "(# *= #)",header : "arrayfire.h".}
proc `/=`*(this: var Matrix; val: Matrix) {.cdecl, importcpp: "(# /= #)", header : "arrayfire.h".}
proc `/=`*(this: var Matrix; val: cdouble) {.cdecl, importcpp: "(# /= #)", header : "arrayfire.h".}
proc `/=`*(this: var Matrix; val: cfloat) {.cdecl, importcpp: "(# /= #)", header : "arrayfire.h".}
proc `/=`*(this: var Matrix; val: cint) {.cdecl, importcpp: "(# /= #)", header : "arrayfire.h".}
proc `/=`*(this: var Matrix; val: cuint) {.cdecl, importcpp: "(# /= #)", header : "arrayfire.h".}
proc `/=`*(this: var Matrix; val: bool) {.cdecl, importcpp: "(# /= #)", header : "arrayfire.h".}
proc `/=`*(this: var Matrix; val: char) {.cdecl, importcpp: "(# /= #)", header : "arrayfire.h".}
proc `/=`*(this: var Matrix; val: clong) {.cdecl, importcpp: "(# /= #)", header : "arrayfire.h".}
proc `/=`*(this: var Matrix; val: culong) {.cdecl, importcpp: "(# /= #)", header : "arrayfire.h".}
proc `/=`*(this: var Matrix; val: clonglong) {.cdecl, importcpp: "(# /= #)",header : "arrayfire.h".}
proc `/=`*(this: var Matrix; val: culonglong) {.cdecl, importcpp: "(# /= #)",header : "arrayfire.h".}
proc `-`*(this: Matrix): Matrix {.noSideEffect, cdecl, importcpp: "(- #)", header : "arrayfire.h".}
proc `!`*(this: Matrix): Matrix {.noSideEffect, cdecl, importcpp: "(! #)", header : "arrayfire.h".}
proc nonzeros*(this: Matrix): cint {.noSideEffect, cdecl, importcpp: "nonzeros",header : "arrayfire.h".}
proc lock*(this: Matrix) {.noSideEffect, cdecl, importcpp: "lock", header : "arrayfire.h".}
proc unlock*(this: Matrix) {.noSideEffect, cdecl, importcpp: "unlock", header : "arrayfire.h".}
proc `+`*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
proc `+`*(lhs: bool; rhs: Matrix): Matrix {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
proc `+`*(lhs: cint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
proc `+`*(lhs: cuint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
proc `+`*(lhs: char; rhs: Matrix): Matrix {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
proc `+`*(lhs: clong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
proc `+`*(lhs: culong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
proc `+`*(lhs: clonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# + #)",header : "arrayfire.h".}
proc `+`*(lhs: culonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# + #)",header : "arrayfire.h".}
proc `+`*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
proc `+`*(lhs: cfloat; rhs: Matrix): Matrix {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
proc `+`*(lhs: Matrix; rhs: bool): Matrix {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
proc `+`*(lhs: Matrix; rhs: cint): Matrix {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
proc `+`*(lhs: Matrix; rhs: cuint): Matrix {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
proc `+`*(lhs: Matrix; rhs: char): Matrix {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
proc `+`*(lhs: Matrix; rhs: clong): Matrix {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
proc `+`*(lhs: Matrix; rhs: culong): Matrix {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
proc `+`*(lhs: Matrix; rhs: clonglong): Matrix {.cdecl, importcpp: "(# + #)",header : "arrayfire.h".}
proc `+`*(lhs: Matrix; rhs: culonglong): Matrix {.cdecl, importcpp: "(# + #)",header : "arrayfire.h".}
proc `+`*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
proc `+`*(lhs: Matrix; rhs: cfloat): Matrix {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
proc `-`*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}
proc `-`*(lhs: bool; rhs: Matrix): Matrix {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}
proc `-`*(lhs: cint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}
proc `-`*(lhs: cuint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}
proc `-`*(lhs: char; rhs: Matrix): Matrix {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}
proc `-`*(lhs: clong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}
proc `-`*(lhs: culong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}
proc `-`*(lhs: clonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# - #)",header : "arrayfire.h".}
proc `-`*(lhs: culonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# - #)",header : "arrayfire.h".}
proc `-`*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}
proc `-`*(lhs: cfloat; rhs: Matrix): Matrix {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}
proc `-`*(lhs: Matrix; rhs: bool): Matrix {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}
proc `-`*(lhs: Matrix; rhs: cint): Matrix {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}
proc `-`*(lhs: Matrix; rhs: cuint): Matrix {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}
proc `-`*(lhs: Matrix; rhs: char): Matrix {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}
proc `-`*(lhs: Matrix; rhs: clong): Matrix {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}
proc `-`*(lhs: Matrix; rhs: culong): Matrix {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}
proc `-`*(lhs: Matrix; rhs: clonglong): Matrix {.cdecl, importcpp: "(# - #)",header : "arrayfire.h".}
proc `-`*(lhs: Matrix; rhs: culonglong): Matrix {.cdecl, importcpp: "(# - #)",header : "arrayfire.h".}
proc `-`*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}
proc `-`*(lhs: Matrix; rhs: cfloat): Matrix {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}
proc `*`*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}
proc `*`*(lhs: bool; rhs: Matrix): Matrix {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}
proc `*`*(lhs: cint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}
proc `*`*(lhs: cuint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}
proc `*`*(lhs: char; rhs: Matrix): Matrix {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}
proc `*`*(lhs: clong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}
proc `*`*(lhs: culong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}
proc `*`*(lhs: clonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# * #)",header : "arrayfire.h".}
proc `*`*(lhs: culonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# * #)",header : "arrayfire.h".}
proc `*`*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}
proc `*`*(lhs: cfloat; rhs: Matrix): Matrix {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}
proc `*`*(lhs: Matrix; rhs: bool): Matrix {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}
proc `*`*(lhs: Matrix; rhs: cint): Matrix {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}
proc `*`*(lhs: Matrix; rhs: cuint): Matrix {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}
proc `*`*(lhs: Matrix; rhs: char): Matrix {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}
proc `*`*(lhs: Matrix; rhs: clong): Matrix {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}
proc `*`*(lhs: Matrix; rhs: culong): Matrix {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}
proc `*`*(lhs: Matrix; rhs: clonglong): Matrix {.cdecl, importcpp: "(# * #)",header : "arrayfire.h".}
proc `*`*(lhs: Matrix; rhs: culonglong): Matrix {.cdecl, importcpp: "(# * #)",header : "arrayfire.h".}
proc `*`*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}
proc `*`*(lhs: Matrix; rhs: cfloat): Matrix {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}
proc `/`*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
proc `/`*(lhs: bool; rhs: Matrix): Matrix {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
proc `/`*(lhs: cint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
proc `/`*(lhs: cuint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
proc `/`*(lhs: char; rhs: Matrix): Matrix {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
proc `/`*(lhs: clong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
proc `/`*(lhs: culong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
proc `/`*(lhs: clonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# / #)",header : "arrayfire.h".}
proc `/`*(lhs: culonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# / #)",header : "arrayfire.h".}
proc `/`*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
proc `/`*(lhs: cfloat; rhs: Matrix): Matrix {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
proc `/`*(lhs: Matrix; rhs: bool): Matrix {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
proc `/`*(lhs: Matrix; rhs: cint): Matrix {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
proc `/`*(lhs: Matrix; rhs: cuint): Matrix {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
proc `/`*(lhs: Matrix; rhs: char): Matrix {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
proc `/`*(lhs: Matrix; rhs: clong): Matrix {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
proc `/`*(lhs: Matrix; rhs: culong): Matrix {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
proc `/`*(lhs: Matrix; rhs: clonglong): Matrix {.cdecl, importcpp: "(# / #)",header : "arrayfire.h".}
proc `/`*(lhs: Matrix; rhs: culonglong): Matrix {.cdecl, importcpp: "(# / #)",header : "arrayfire.h".}
proc `/`*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
proc `/`*(lhs: Matrix; rhs: cfloat): Matrix {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
proc `==`*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "(# == #)",header : "arrayfire.h".}
proc `==`*(lhs: bool; rhs: Matrix): Matrix {.cdecl, importcpp: "(# == #)", header : "arrayfire.h".}
proc `==`*(lhs: cint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# == #)", header : "arrayfire.h".}
proc `==`*(lhs: cuint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# == #)", header : "arrayfire.h".}
proc `==`*(lhs: char; rhs: Matrix): Matrix {.cdecl, importcpp: "(# == #)", header : "arrayfire.h".}
proc `==`*(lhs: clong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# == #)", header : "arrayfire.h".}
proc `==`*(lhs: culong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# == #)",header : "arrayfire.h".}
proc `==`*(lhs: clonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# == #)",header : "arrayfire.h".}
proc `==`*(lhs: culonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# == #)",header : "arrayfire.h".}
proc `==`*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "(# == #)",header : "arrayfire.h".}
proc `==`*(lhs: cfloat; rhs: Matrix): Matrix {.cdecl, importcpp: "(# == #)",header : "arrayfire.h".}
proc `==`*(lhs: Matrix; rhs: bool): Matrix {.cdecl, importcpp: "(# == #)", header : "arrayfire.h".}
proc `==`*(lhs: Matrix; rhs: cint): Matrix {.cdecl, importcpp: "(# == #)", header : "arrayfire.h".}
proc `==`*(lhs: Matrix; rhs: cuint): Matrix {.cdecl, importcpp: "(# == #)", header : "arrayfire.h".}
proc `==`*(lhs: Matrix; rhs: char): Matrix {.cdecl, importcpp: "(# == #)", header : "arrayfire.h".}
proc `==`*(lhs: Matrix; rhs: clong): Matrix {.cdecl, importcpp: "(# == #)", header : "arrayfire.h".}
proc `==`*(lhs: Matrix; rhs: culong): Matrix {.cdecl, importcpp: "(# == #)",header : "arrayfire.h".}
proc `==`*(lhs: Matrix; rhs: clonglong): Matrix {.cdecl, importcpp: "(# == #)",header : "arrayfire.h".}
proc `==`*(lhs: Matrix; rhs: culonglong): Matrix {.cdecl, importcpp: "(# == #)",header : "arrayfire.h".}
proc `==`*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "(# == #)",header : "arrayfire.h".}
proc `==`*(lhs: Matrix; rhs: cfloat): Matrix {.cdecl, importcpp: "(# == #)",header : "arrayfire.h".}
proc `<`*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "(# < #)", header : "arrayfire.h".}
proc `<`*(lhs: bool; rhs: Matrix): Matrix {.cdecl, importcpp: "(# < #)", header : "arrayfire.h".}
proc `<`*(lhs: cint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# < #)", header : "arrayfire.h".}
proc `<`*(lhs: cuint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# < #)", header : "arrayfire.h".}
proc `<`*(lhs: char; rhs: Matrix): Matrix {.cdecl, importcpp: "(# < #)", header : "arrayfire.h".}
proc `<`*(lhs: clong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# < #)", header : "arrayfire.h".}
proc `<`*(lhs: culong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# < #)", header : "arrayfire.h".}
proc `<`*(lhs: clonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# < #)",header : "arrayfire.h".}
proc `<`*(lhs: culonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# < #)",header : "arrayfire.h".}
proc `<`*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "(# < #)", header : "arrayfire.h".}
proc `<`*(lhs: cfloat; rhs: Matrix): Matrix {.cdecl, importcpp: "(# < #)", header : "arrayfire.h".}
proc `<`*(lhs: Matrix; rhs: bool): Matrix {.cdecl, importcpp: "(# < #)", header : "arrayfire.h".}
proc `<`*(lhs: Matrix; rhs: cint): Matrix {.cdecl, importcpp: "(# < #)", header : "arrayfire.h".}
proc `<`*(lhs: Matrix; rhs: cuint): Matrix {.cdecl, importcpp: "(# < #)", header : "arrayfire.h".}
proc `<`*(lhs: Matrix; rhs: char): Matrix {.cdecl, importcpp: "(# < #)", header : "arrayfire.h".}
proc `<`*(lhs: Matrix; rhs: clong): Matrix {.cdecl, importcpp: "(# < #)", header : "arrayfire.h".}
proc `<`*(lhs: Matrix; rhs: culong): Matrix {.cdecl, importcpp: "(# < #)", header : "arrayfire.h".}
proc `<`*(lhs: Matrix; rhs: clonglong): Matrix {.cdecl, importcpp: "(# < #)",header : "arrayfire.h".}
proc `<`*(lhs: Matrix; rhs: culonglong): Matrix {.cdecl, importcpp: "(# < #)",header : "arrayfire.h".}
proc `<`*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "(# < #)", header : "arrayfire.h".}
proc `<`*(lhs: Matrix; rhs: cfloat): Matrix {.cdecl, importcpp: "(# < #)", header : "arrayfire.h".}
proc `<=`*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "(# <= #)",header : "arrayfire.h".}
proc `<=`*(lhs: bool; rhs: Matrix): Matrix {.cdecl, importcpp: "(# <= #)", header : "arrayfire.h".}
proc `<=`*(lhs: cint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# <= #)", header : "arrayfire.h".}
proc `<=`*(lhs: cuint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# <= #)", header : "arrayfire.h".}
proc `<=`*(lhs: char; rhs: Matrix): Matrix {.cdecl, importcpp: "(# <= #)", header : "arrayfire.h".}
proc `<=`*(lhs: clong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# <= #)", header : "arrayfire.h".}
proc `<=`*(lhs: culong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# <= #)",header : "arrayfire.h".}
proc `<=`*(lhs: clonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# <= #)",header : "arrayfire.h".}
proc `<=`*(lhs: culonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# <= #)",header : "arrayfire.h".}
proc `<=`*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "(# <= #)",header : "arrayfire.h".}
proc `<=`*(lhs: cfloat; rhs: Matrix): Matrix {.cdecl, importcpp: "(# <= #)",header : "arrayfire.h".}
proc `<=`*(lhs: Matrix; rhs: bool): Matrix {.cdecl, importcpp: "(# <= #)", header : "arrayfire.h".}
proc `<=`*(lhs: Matrix; rhs: cint): Matrix {.cdecl, importcpp: "(# <= #)", header : "arrayfire.h".}
proc `<=`*(lhs: Matrix; rhs: cuint): Matrix {.cdecl, importcpp: "(# <= #)", header : "arrayfire.h".}
proc `<=`*(lhs: Matrix; rhs: char): Matrix {.cdecl, importcpp: "(# <= #)", header : "arrayfire.h".}
proc `<=`*(lhs: Matrix; rhs: clong): Matrix {.cdecl, importcpp: "(# <= #)", header : "arrayfire.h".}
proc `<=`*(lhs: Matrix; rhs: culong): Matrix {.cdecl, importcpp: "(# <= #)",header : "arrayfire.h".}
proc `<=`*(lhs: Matrix; rhs: clonglong): Matrix {.cdecl, importcpp: "(# <= #)",header : "arrayfire.h".}
proc `<=`*(lhs: Matrix; rhs: culonglong): Matrix {.cdecl, importcpp: "(# <= #)",header : "arrayfire.h".}
proc `<=`*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "(# <= #)",header : "arrayfire.h".}
proc `<=`*(lhs: Matrix; rhs: cfloat): Matrix {.cdecl, importcpp: "(# <= #)",header : "arrayfire.h".}
proc `&&`*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "(# && #)",header : "arrayfire.h".}
proc `&&`*(lhs: bool; rhs: Matrix): Matrix {.cdecl, importcpp: "(# && #)", header : "arrayfire.h".}
proc `&&`*(lhs: cint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# && #)", header : "arrayfire.h".}
proc `&&`*(lhs: cuint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# && #)", header : "arrayfire.h".}
proc `&&`*(lhs: char; rhs: Matrix): Matrix {.cdecl, importcpp: "(# && #)", header : "arrayfire.h".}
proc `&&`*(lhs: clong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# && #)", header : "arrayfire.h".}
proc `&&`*(lhs: culong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# && #)",header : "arrayfire.h".}
proc `&&`*(lhs: clonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# && #)",header : "arrayfire.h".}
proc `&&`*(lhs: culonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# && #)",header : "arrayfire.h".}
proc `&&`*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "(# && #)",header : "arrayfire.h".}
proc `&&`*(lhs: cfloat; rhs: Matrix): Matrix {.cdecl, importcpp: "(# && #)",header : "arrayfire.h".}
proc `&&`*(lhs: Matrix; rhs: bool): Matrix {.cdecl, importcpp: "(# && #)", header : "arrayfire.h".}
proc `&&`*(lhs: Matrix; rhs: cint): Matrix {.cdecl, importcpp: "(# && #)", header : "arrayfire.h".}
proc `&&`*(lhs: Matrix; rhs: cuint): Matrix {.cdecl, importcpp: "(# && #)", header : "arrayfire.h".}
proc `&&`*(lhs: Matrix; rhs: char): Matrix {.cdecl, importcpp: "(# && #)", header : "arrayfire.h".}
proc `&&`*(lhs: Matrix; rhs: clong): Matrix {.cdecl, importcpp: "(# && #)", header : "arrayfire.h".}
proc `&&`*(lhs: Matrix; rhs: culong): Matrix {.cdecl, importcpp: "(# && #)",
    header : "arrayfire.h".}
proc `&&`*(lhs: Matrix; rhs: clonglong): Matrix {.cdecl, importcpp: "(# && #)",
    header : "arrayfire.h".}
proc `&&`*(lhs: Matrix; rhs: culonglong): Matrix {.cdecl, importcpp: "(# && #)",
    header : "arrayfire.h".}
proc `&&`*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "(# && #)",
    header : "arrayfire.h".}
proc `&&`*(lhs: Matrix; rhs: cfloat): Matrix {.cdecl, importcpp: "(# && #)",
    header : "arrayfire.h".}
proc `||`*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "(# || #)",
    header : "arrayfire.h".}
proc `||`*(lhs: bool; rhs: Matrix): Matrix {.cdecl, importcpp: "(# || #)", header : "arrayfire.h".}
proc `||`*(lhs: cint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# || #)", header : "arrayfire.h".}
proc `||`*(lhs: cuint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# || #)", header : "arrayfire.h".}
proc `||`*(lhs: char; rhs: Matrix): Matrix {.cdecl, importcpp: "(# || #)", header : "arrayfire.h".}
proc `||`*(lhs: clong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# || #)", header : "arrayfire.h".}
proc `||`*(lhs: culong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# || #)",
    header : "arrayfire.h".}
proc `||`*(lhs: clonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# || #)",
    header : "arrayfire.h".}
proc `||`*(lhs: culonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# || #)",
    header : "arrayfire.h".}
proc `||`*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "(# || #)",
    header : "arrayfire.h".}
proc `||`*(lhs: cfloat; rhs: Matrix): Matrix {.cdecl, importcpp: "(# || #)",
    header : "arrayfire.h".}
proc `||`*(lhs: Matrix; rhs: bool): Matrix {.cdecl, importcpp: "(# || #)", header : "arrayfire.h".}
proc `||`*(lhs: Matrix; rhs: cint): Matrix {.cdecl, importcpp: "(# || #)", header : "arrayfire.h".}
proc `||`*(lhs: Matrix; rhs: cuint): Matrix {.cdecl, importcpp: "(# || #)", header : "arrayfire.h".}
proc `||`*(lhs: Matrix; rhs: char): Matrix {.cdecl, importcpp: "(# || #)", header : "arrayfire.h".}
proc `||`*(lhs: Matrix; rhs: clong): Matrix {.cdecl, importcpp: "(# || #)", header : "arrayfire.h".}
proc `||`*(lhs: Matrix; rhs: culong): Matrix {.cdecl, importcpp: "(# || #)",
    header : "arrayfire.h".}
proc `||`*(lhs: Matrix; rhs: clonglong): Matrix {.cdecl, importcpp: "(# || #)",
    header : "arrayfire.h".}
proc `||`*(lhs: Matrix; rhs: culonglong): Matrix {.cdecl, importcpp: "(# || #)",
    header : "arrayfire.h".}
proc `||`*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "(# || #)",
    header : "arrayfire.h".}
proc `||`*(lhs: Matrix; rhs: cfloat): Matrix {.cdecl, importcpp: "(# || #)",
    header : "arrayfire.h".}
proc `%`*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "(# % #)", header : "arrayfire.h".}
proc `%`*(lhs: bool; rhs: Matrix): Matrix {.cdecl, importcpp: "(# % #)", header : "arrayfire.h".}
proc `%`*(lhs: cint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# % #)", header : "arrayfire.h".}
proc `%`*(lhs: cuint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# % #)", header : "arrayfire.h".}
proc `%`*(lhs: char; rhs: Matrix): Matrix {.cdecl, importcpp: "(# % #)", header : "arrayfire.h".}
proc `%`*(lhs: clong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# % #)", header : "arrayfire.h".}
proc `%`*(lhs: culong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# % #)", header : "arrayfire.h".}
proc `%`*(lhs: clonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# % #)",
    header : "arrayfire.h".}
proc `%`*(lhs: culonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# % #)",
    header : "arrayfire.h".}
proc `%`*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "(# % #)", header : "arrayfire.h".}
proc `%`*(lhs: cfloat; rhs: Matrix): Matrix {.cdecl, importcpp: "(# % #)", header : "arrayfire.h".}
proc `%`*(lhs: Matrix; rhs: bool): Matrix {.cdecl, importcpp: "(# % #)", header : "arrayfire.h".}
proc `%`*(lhs: Matrix; rhs: cint): Matrix {.cdecl, importcpp: "(# % #)", header : "arrayfire.h".}
proc `%`*(lhs: Matrix; rhs: cuint): Matrix {.cdecl, importcpp: "(# % #)", header : "arrayfire.h".}
proc `%`*(lhs: Matrix; rhs: char): Matrix {.cdecl, importcpp: "(# % #)", header : "arrayfire.h".}
proc `%`*(lhs: Matrix; rhs: clong): Matrix {.cdecl, importcpp: "(# % #)", header : "arrayfire.h".}
proc `%`*(lhs: Matrix; rhs: culong): Matrix {.cdecl, importcpp: "(# % #)", header : "arrayfire.h".}
proc `%`*(lhs: Matrix; rhs: clonglong): Matrix {.cdecl, importcpp: "(# % #)",
    header : "arrayfire.h".}
proc `%`*(lhs: Matrix; rhs: culonglong): Matrix {.cdecl, importcpp: "(# % #)",
    header : "arrayfire.h".}
proc `%`*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "(# % #)", header : "arrayfire.h".}
proc `%`*(lhs: Matrix; rhs: cfloat): Matrix {.cdecl, importcpp: "(# % #)", header : "arrayfire.h".}
proc `&`*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "(# & #)", header : "arrayfire.h".}
proc `&`*(lhs: bool; rhs: Matrix): Matrix {.cdecl, importcpp: "(# & #)", header : "arrayfire.h".}
proc `&`*(lhs: cint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# & #)", header : "arrayfire.h".}
proc `&`*(lhs: cuint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# & #)", header : "arrayfire.h".}
proc `&`*(lhs: char; rhs: Matrix): Matrix {.cdecl, importcpp: "(# & #)", header : "arrayfire.h".}
proc `&`*(lhs: clong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# & #)", header : "arrayfire.h".}
proc `&`*(lhs: culong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# & #)", header : "arrayfire.h".}
proc `&`*(lhs: clonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# & #)",
    header : "arrayfire.h".}
proc `&`*(lhs: culonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# & #)",
    header : "arrayfire.h".}
proc `&`*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "(# & #)", header : "arrayfire.h".}
proc `&`*(lhs: cfloat; rhs: Matrix): Matrix {.cdecl, importcpp: "(# & #)", header : "arrayfire.h".}
proc `&`*(lhs: Matrix; rhs: bool): Matrix {.cdecl, importcpp: "(# & #)", header : "arrayfire.h".}
proc `&`*(lhs: Matrix; rhs: cint): Matrix {.cdecl, importcpp: "(# & #)", header : "arrayfire.h".}
proc `&`*(lhs: Matrix; rhs: cuint): Matrix {.cdecl, importcpp: "(# & #)", header : "arrayfire.h".}
proc `&`*(lhs: Matrix; rhs: char): Matrix {.cdecl, importcpp: "(# & #)", header : "arrayfire.h".}
proc `&`*(lhs: Matrix; rhs: clong): Matrix {.cdecl, importcpp: "(# & #)", header : "arrayfire.h".}
proc `&`*(lhs: Matrix; rhs: culong): Matrix {.cdecl, importcpp: "(# & #)", header : "arrayfire.h".}
proc `&`*(lhs: Matrix; rhs: clonglong): Matrix {.cdecl, importcpp: "(# & #)",
    header : "arrayfire.h".}
proc `&`*(lhs: Matrix; rhs: culonglong): Matrix {.cdecl, importcpp: "(# & #)",
    header : "arrayfire.h".}
proc `&`*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "(# & #)", header : "arrayfire.h".}
proc `&`*(lhs: Matrix; rhs: cfloat): Matrix {.cdecl, importcpp: "(# & #)", header : "arrayfire.h".}
proc `|`*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "(# | #)", header : "arrayfire.h".}
proc `|`*(lhs: bool; rhs: Matrix): Matrix {.cdecl, importcpp: "(# | #)", header : "arrayfire.h".}
proc `|`*(lhs: cint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# | #)", header : "arrayfire.h".}
proc `|`*(lhs: cuint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# | #)", header : "arrayfire.h".}
proc `|`*(lhs: char; rhs: Matrix): Matrix {.cdecl, importcpp: "(# | #)", header : "arrayfire.h".}
proc `|`*(lhs: clong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# | #)", header : "arrayfire.h".}
proc `|`*(lhs: culong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# | #)", header : "arrayfire.h".}
proc `|`*(lhs: clonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# | #)",
    header : "arrayfire.h".}
proc `|`*(lhs: culonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# | #)",
    header : "arrayfire.h".}
proc `|`*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "(# | #)", header : "arrayfire.h".}
proc `|`*(lhs: cfloat; rhs: Matrix): Matrix {.cdecl, importcpp: "(# | #)", header : "arrayfire.h".}
proc `|`*(lhs: Matrix; rhs: bool): Matrix {.cdecl, importcpp: "(# | #)", header : "arrayfire.h".}
proc `|`*(lhs: Matrix; rhs: cint): Matrix {.cdecl, importcpp: "(# | #)", header : "arrayfire.h".}
proc `|`*(lhs: Matrix; rhs: cuint): Matrix {.cdecl, importcpp: "(# | #)", header : "arrayfire.h".}
proc `|`*(lhs: Matrix; rhs: char): Matrix {.cdecl, importcpp: "(# | #)", header : "arrayfire.h".}
proc `|`*(lhs: Matrix; rhs: clong): Matrix {.cdecl, importcpp: "(# | #)", header : "arrayfire.h".}
proc `|`*(lhs: Matrix; rhs: culong): Matrix {.cdecl, importcpp: "(# | #)", header : "arrayfire.h".}
proc `|`*(lhs: Matrix; rhs: clonglong): Matrix {.cdecl, importcpp: "(# | #)",
    header : "arrayfire.h".}
proc `|`*(lhs: Matrix; rhs: culonglong): Matrix {.cdecl, importcpp: "(# | #)",
    header : "arrayfire.h".}
proc `|`*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "(# | #)", header : "arrayfire.h".}
proc `|`*(lhs: Matrix; rhs: cfloat): Matrix {.cdecl, importcpp: "(# | #)", header : "arrayfire.h".}
proc `^`*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "(# ^ #)", header : "arrayfire.h".}
proc `^`*(lhs: bool; rhs: Matrix): Matrix {.cdecl, importcpp: "(# ^ #)", header : "arrayfire.h".}
proc `^`*(lhs: cint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# ^ #)", header : "arrayfire.h".}
proc `^`*(lhs: cuint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# ^ #)", header : "arrayfire.h".}
proc `^`*(lhs: char; rhs: Matrix): Matrix {.cdecl, importcpp: "(# ^ #)", header : "arrayfire.h".}
proc `^`*(lhs: clong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# ^ #)", header : "arrayfire.h".}
proc `^`*(lhs: culong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# ^ #)", header : "arrayfire.h".}
proc `^`*(lhs: clonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# ^ #)",
    header : "arrayfire.h".}
proc `^`*(lhs: culonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# ^ #)",
    header : "arrayfire.h".}
proc `^`*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "(# ^ #)", header : "arrayfire.h".}
proc `^`*(lhs: cfloat; rhs: Matrix): Matrix {.cdecl, importcpp: "(# ^ #)", header : "arrayfire.h".}
proc `^`*(lhs: Matrix; rhs: bool): Matrix {.cdecl, importcpp: "(# ^ #)", header : "arrayfire.h".}
proc `^`*(lhs: Matrix; rhs: cint): Matrix {.cdecl, importcpp: "(# ^ #)", header : "arrayfire.h".}
proc `^`*(lhs: Matrix; rhs: cuint): Matrix {.cdecl, importcpp: "(# ^ #)", header : "arrayfire.h".}
proc `^`*(lhs: Matrix; rhs: char): Matrix {.cdecl, importcpp: "(# ^ #)", header : "arrayfire.h".}
proc `^`*(lhs: Matrix; rhs: clong): Matrix {.cdecl, importcpp: "(# ^ #)", header : "arrayfire.h".}
proc `^`*(lhs: Matrix; rhs: culong): Matrix {.cdecl, importcpp: "(# ^ #)", header : "arrayfire.h".}
proc `^`*(lhs: Matrix; rhs: clonglong): Matrix {.cdecl, importcpp: "(# ^ #)",
    header : "arrayfire.h".}
proc `^`*(lhs: Matrix; rhs: culonglong): Matrix {.cdecl, importcpp: "(# ^ #)",
    header : "arrayfire.h".}
proc `^`*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "(# ^ #)", header : "arrayfire.h".}
proc `^`*(lhs: Matrix; rhs: cfloat): Matrix {.cdecl, importcpp: "(# ^ #)", header : "arrayfire.h".}
proc `<<`*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "(# << #)",
    header : "arrayfire.h".}
proc `<<`*(lhs: bool; rhs: Matrix): Matrix {.cdecl, importcpp: "(# << #)", header : "arrayfire.h".}
proc `<<`*(lhs: cint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# << #)", header : "arrayfire.h".}
proc `<<`*(lhs: cuint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# << #)", header : "arrayfire.h".}
proc `<<`*(lhs: char; rhs: Matrix): Matrix {.cdecl, importcpp: "(# << #)", header : "arrayfire.h".}
proc `<<`*(lhs: clong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# << #)", header : "arrayfire.h".}
proc `<<`*(lhs: culong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# << #)",
    header : "arrayfire.h".}
proc `<<`*(lhs: clonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# << #)",
    header : "arrayfire.h".}
proc `<<`*(lhs: culonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# << #)",
    header : "arrayfire.h".}
proc `<<`*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "(# << #)",
    header : "arrayfire.h".}
proc `<<`*(lhs: cfloat; rhs: Matrix): Matrix {.cdecl, importcpp: "(# << #)",
    header : "arrayfire.h".}
proc `<<`*(lhs: Matrix; rhs: bool): Matrix {.cdecl, importcpp: "(# << #)", header : "arrayfire.h".}
proc `<<`*(lhs: Matrix; rhs: cint): Matrix {.cdecl, importcpp: "(# << #)", header : "arrayfire.h".}
proc `<<`*(lhs: Matrix; rhs: cuint): Matrix {.cdecl, importcpp: "(# << #)", header : "arrayfire.h".}
proc `<<`*(lhs: Matrix; rhs: char): Matrix {.cdecl, importcpp: "(# << #)", header : "arrayfire.h".}
proc `<<`*(lhs: Matrix; rhs: clong): Matrix {.cdecl, importcpp: "(# << #)", header : "arrayfire.h".}
proc `<<`*(lhs: Matrix; rhs: culong): Matrix {.cdecl, importcpp: "(# << #)",
    header : "arrayfire.h".}
proc `<<`*(lhs: Matrix; rhs: clonglong): Matrix {.cdecl, importcpp: "(# << #)",
    header : "arrayfire.h".}
proc `<<`*(lhs: Matrix; rhs: culonglong): Matrix {.cdecl, importcpp: "(# << #)",
    header : "arrayfire.h".}
proc `<<`*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "(# << #)",
    header : "arrayfire.h".}
proc `<<`*(lhs: Matrix; rhs: cfloat): Matrix {.cdecl, importcpp: "(# << #)",
    header : "arrayfire.h".}
proc `>>`*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "(# >> #)",
    header : "arrayfire.h".}
proc `>>`*(lhs: bool; rhs: Matrix): Matrix {.cdecl, importcpp: "(# >> #)", header : "arrayfire.h".}
proc `>>`*(lhs: cint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# >> #)", header : "arrayfire.h".}
proc `>>`*(lhs: cuint; rhs: Matrix): Matrix {.cdecl, importcpp: "(# >> #)", header : "arrayfire.h".}
proc `>>`*(lhs: char; rhs: Matrix): Matrix {.cdecl, importcpp: "(# >> #)", header : "arrayfire.h".}
proc `>>`*(lhs: clong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# >> #)", header : "arrayfire.h".}
proc `>>`*(lhs: culong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# >> #)",
    header : "arrayfire.h".}
proc `>>`*(lhs: clonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# >> #)",
    header : "arrayfire.h".}
proc `>>`*(lhs: culonglong; rhs: Matrix): Matrix {.cdecl, importcpp: "(# >> #)",
    header : "arrayfire.h".}
proc `>>`*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "(# >> #)",
    header : "arrayfire.h".}
proc `>>`*(lhs: cfloat; rhs: Matrix): Matrix {.cdecl, importcpp: "(# >> #)",
    header : "arrayfire.h".}
proc `>>`*(lhs: Matrix; rhs: bool): Matrix {.cdecl, importcpp: "(# >> #)", header : "arrayfire.h".}
proc `>>`*(lhs: Matrix; rhs: cint): Matrix {.cdecl, importcpp: "(# >> #)", header : "arrayfire.h".}
proc `>>`*(lhs: Matrix; rhs: cuint): Matrix {.cdecl, importcpp: "(# >> #)", header : "arrayfire.h".}
proc `>>`*(lhs: Matrix; rhs: char): Matrix {.cdecl, importcpp: "(# >> #)", header : "arrayfire.h".}
proc `>>`*(lhs: Matrix; rhs: clong): Matrix {.cdecl, importcpp: "(# >> #)", header : "arrayfire.h".}
proc `>>`*(lhs: Matrix; rhs: culong): Matrix {.cdecl, importcpp: "(# >> #)",
    header : "arrayfire.h".}
proc `>>`*(lhs: Matrix; rhs: clonglong): Matrix {.cdecl, importcpp: "(# >> #)",
    header : "arrayfire.h".}
proc `>>`*(lhs: Matrix; rhs: culonglong): Matrix {.cdecl, importcpp: "(# >> #)",
    header : "arrayfire.h".}
proc `>>`*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "(# >> #)",
    header : "arrayfire.h".}
proc `>>`*(lhs: Matrix; rhs: cfloat): Matrix {.cdecl, importcpp: "(# >> #)",
    header : "arrayfire.h".}

proc min*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "min(@)", header : "arrayfire.h".}

proc min*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "min(@)", header : "arrayfire.h".}

proc min*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "min(@)", header : "arrayfire.h".}

proc max*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "max(@)", header : "arrayfire.h".}

proc max*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "max(@)", header : "arrayfire.h".}

proc max*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "max(@)", header : "arrayfire.h".}

proc clamp*(matin : Matrix; lo: Matrix; hi: Matrix): Matrix {.cdecl, importcpp: "clamp(@)",
    header : "arrayfire.h".}

proc clamp*(matin : Matrix; lo: Matrix; hi: cdouble): Matrix {.cdecl,
    importcpp: "clamp(@)", header : "arrayfire.h".}

proc clamp*(matin : Matrix; lo: cdouble; hi: Matrix): Matrix {.cdecl,
    importcpp: "clamp(@)", header : "arrayfire.h".}

proc clamp*(matin : Matrix; lo: cdouble; hi: cdouble): Matrix {.cdecl,
    importcpp: "clamp(@)", header : "arrayfire.h".}

proc rem*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "rem(@)", header : "arrayfire.h".}

proc rem*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "rem(@)", header : "arrayfire.h".}

proc rem*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "rem(@)", header : "arrayfire.h".}

proc `mod`*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "mod(@)", header : "arrayfire.h".}

proc `mod`*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "mod(@)",
    header : "arrayfire.h".}

proc `mod`*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "mod(@)",
    header : "arrayfire.h".}

proc pad*(matin: Matrix; beginPadding: Dim4, endPadding: Dim4, padFillType: BorderType ): Matrix {.cdecl, importcpp: "pad(@)",
    header : "arrayfire.h".}

proc abs*(matin : Matrix): Matrix {.cdecl, importcpp: "abs(@)", header : "arrayfire.h".}

proc arg*(matin : Matrix): Matrix {.cdecl, importcpp: "arg(@)", header : "arrayfire.h".}

proc sign*(matin : Matrix): Matrix {.cdecl, importcpp: "sign(@)", header : "arrayfire.h".}

proc round*(matin : Matrix): Matrix {.cdecl, importcpp: "round(@)", header : "arrayfire.h".}

proc trunc*(matin : Matrix): Matrix {.cdecl, importcpp: "trunc(@)", header : "arrayfire.h".}

proc floor*(matin : Matrix): Matrix {.cdecl, importcpp: "floor(@)", header : "arrayfire.h".}

proc ceil*(matin : Matrix): Matrix {.cdecl, importcpp: "ceil(@)", header : "arrayfire.h".}

proc hypot*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "hypot(@)",
    header : "arrayfire.h".}

proc hypot*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "hypot(@)",
    header : "arrayfire.h".}

proc hypot*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "hypot(@)",
    header : "arrayfire.h".}

proc sin*(matin : Matrix): Matrix {.cdecl, importcpp: "sin(@)", header : "arrayfire.h".}

proc cos*(matin : Matrix): Matrix {.cdecl, importcpp: "cos(@)", header : "arrayfire.h".}

proc tan*(matin : Matrix): Matrix {.cdecl, importcpp: "tan(@)", header : "arrayfire.h".}

proc asin*(matin : Matrix): Matrix {.cdecl, importcpp: "asin(@)", header : "arrayfire.h".}

proc acos*(matin : Matrix): Matrix {.cdecl, importcpp: "acos(@)", header : "arrayfire.h".}

proc atan*(matin : Matrix): Matrix {.cdecl, importcpp: "atan(@)", header : "arrayfire.h".}

proc atan2*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "atan2(@)",
    header : "arrayfire.h".}

proc atan2*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "atan2(@)",
    header : "arrayfire.h".}

proc atan2*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "atan2(@)",
    header : "arrayfire.h".}

proc complex*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "complex(@)",
    header : "arrayfire.h".}

proc complex*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "complex(@)",
    header : "arrayfire.h".}

proc complex*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "complex(@)",
    header : "arrayfire.h".}

proc complex*(matin : Matrix): Matrix {.cdecl, importcpp: "complex(@)", header : "arrayfire.h".}

proc real*(matin : Matrix): Matrix {.cdecl, importcpp: "real(@)", header : "arrayfire.h".}

proc imag*(matin : Matrix): Matrix {.cdecl, importcpp: "imag(@)", header : "arrayfire.h".}

proc conjg*(matin : Matrix): Matrix {.cdecl, importcpp: "conjg(@)", header : "arrayfire.h".}

proc sinh*(matin : Matrix): Matrix {.cdecl, importcpp: "sinh(@)", header : "arrayfire.h".}

proc cosh*(matin : Matrix): Matrix {.cdecl, importcpp: "cosh(@)", header : "arrayfire.h".}

proc tanh*(matin : Matrix): Matrix {.cdecl, importcpp: "tanh(@)", header : "arrayfire.h".}

proc asinh*(matin : Matrix): Matrix {.cdecl, importcpp: "asinh(@)", header : "arrayfire.h".}

proc acosh*(matin : Matrix): Matrix {.cdecl, importcpp: "acosh(@)", header : "arrayfire.h".}

proc atanh*(matin : Matrix): Matrix {.cdecl, importcpp: "atanh(@)", header : "arrayfire.h".}

proc root*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "root(@)", header : "arrayfire.h".}

proc root*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "root(@)",
    header : "arrayfire.h".}

proc root*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "root(@)",
    header : "arrayfire.h".}

proc pow*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "pow(@)", header : "arrayfire.h".}

proc pow*(lhs: Matrix; rhs: cdouble): Matrix {.cdecl, importcpp: "pow(@)", header : "arrayfire.h".}

proc pow*(lhs: cdouble; rhs: Matrix): Matrix {.cdecl, importcpp: "pow(@)", header : "arrayfire.h".}

proc pow2*(matin : Matrix): Matrix {.cdecl, importcpp: "pow2(@)", header : "arrayfire.h".}

proc sigmoid*(matin : Matrix): Matrix {.cdecl, importcpp: "sigmoid(@)", header : "arrayfire.h".}

proc exp*(matin : Matrix): Matrix {.cdecl, importcpp: "exp(@)", header : "arrayfire.h".}

proc expm1*(matin : Matrix): Matrix {.cdecl, importcpp: "expm1(@)", header : "arrayfire.h".}

proc erf*(matin : Matrix): Matrix {.cdecl, importcpp: "erf(@)", header : "arrayfire.h".}

proc erfc*(matin : Matrix): Matrix {.cdecl, importcpp: "erfc(@)", header : "arrayfire.h".}

proc log*(matin : Matrix): Matrix {.cdecl, importcpp: "log(@)", header : "arrayfire.h".}

proc log1p*(matin : Matrix): Matrix {.cdecl, importcpp: "log1p(@)", header : "arrayfire.h".}

proc log10*(matin : Matrix): Matrix {.cdecl, importcpp: "log10(@)", header : "arrayfire.h".}

proc log2*(matin : Matrix): Matrix {.cdecl, importcpp: "log2(@)", header : "arrayfire.h".}

proc sqrt*(matin : Matrix): Matrix {.cdecl, importcpp: "sqrt(@)", header : "arrayfire.h".}

proc rsqrt*(matin : Matrix): Matrix {.cdecl, importcpp: "rsqrt(@)", header : "arrayfire.h".}

proc cbrt*(matin : Matrix): Matrix {.cdecl, importcpp: "cbrt(@)", header : "arrayfire.h".}

proc factorial*(matin : Matrix): Matrix {.cdecl, importcpp: "factorial(@)", header : "arrayfire.h".}

proc tgamma*(matin : Matrix): Matrix {.cdecl, importcpp: "tgamma(@)", header : "arrayfire.h".}

proc lgamma*(matin : Matrix): Matrix {.cdecl, importcpp: "lgamma(@)", header : "arrayfire.h".}

proc iszero*(matin : Matrix): Matrix {.cdecl, importcpp: "iszero(@)", header : "arrayfire.h".}

proc isInf*(matin : Matrix): Matrix {.cdecl, importcpp: "isInf(@)", header : "arrayfire.h".}

proc isNaN*(matin : Matrix): Matrix {.cdecl, importcpp: "isNaN(@)", header : "arrayfire.h".}

proc setBackend*(bknd: Backend) 
  {.cdecl, importcpp: "af::setBackend(@)", header : "arrayfire.h".}

proc getBackendCount*(): cuint 
  {.cdecl, importcpp: "af::getBackendCount(@)", header : "arrayfire.h".}

proc af_getAvailableBackends*(): cint 
  {.cdecl, importcpp: "af::getAvailableBackends(@)", header : "arrayfire.h".}

proc getBackendId*(matin : Matrix): Backend 
  {.cdecl, importcpp: "af::getBackendId(@)", header : "arrayfire.h".}

proc getActiveBackend*(): Backend 
  {.cdecl, importcpp: "af::getActiveBackend(@)",header : "arrayfire.h".}

proc getDeviceId*(matin : Matrix): cint 
  {.cdecl, importcpp: "af::getDeviceId(@)", header : "arrayfire.h".}

#todo: find out why call with all arguments goes to wrong function
proc matmul*(lhs: Matrix; rhs: Matrix; 
             optLhs: MatProp =  MatProp.NONE;
             optRhs: MatProp = MatProp.NONE ): Matrix 
  {.cdecl, importcpp: "af::matmul(#,#)", header : "arrayfire.h".}  

proc matmulNT*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "af::matmulNT(@)",
    header : "arrayfire.h".}

proc matmulTN*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "af::matmulTN(@)",
    header : "arrayfire.h".}

proc matmulTT*(lhs: Matrix; rhs: Matrix): Matrix {.cdecl, importcpp: "af::matmulTT(@)",
    header : "arrayfire.h".}

proc matmul*(a: Matrix; b: Matrix; c: Matrix): Matrix {.cdecl, importcpp: "af::matmul(@)",
    header : "arrayfire.h".}

proc matmul*(a: Matrix; b: Matrix; c: Matrix; d: Matrix): Matrix {.cdecl,
    importcpp: "af::matmul(@)", header : "arrayfire.h".}

proc dot*(lhs: Matrix; rhs: Matrix; optLhs: MatProp ;
         optRhs: MatProp ): Matrix {.cdecl, importcpp: "dot(@)", header : "arrayfire.h".}

proc transpose*(matin : Matrix; conjugate: bool = false): Matrix {.cdecl,
    importcpp: "transpose(@)", header : "arrayfire.h".}

proc transposeInPlace*(`in`: var Matrix; conjugate: bool = false) {.cdecl,
    importcpp: "transposeInPlace(@)", header : "arrayfire.h".}

proc `+`*(lhs: Complex32; rhs: Complex32): Complex32 {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
proc `+`*(lhs: Complex64; rhs: Complex64): Complex64 {.cdecl, importcpp: "(# + #)",
    header : "arrayfire.h".}
proc `-`*(lhs: Complex32; rhs: Complex32): Complex32 {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}
proc `-`*(lhs: Complex64; rhs: Complex64): Complex64 {.cdecl, importcpp: "(# - #)",
    header : "arrayfire.h".}
proc `*`*(lhs: Complex32; rhs: Complex32): Complex32 {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}
proc `*`*(lhs: Complex64; rhs: Complex64): Complex64 {.cdecl, importcpp: "(# * #)",
    header : "arrayfire.h".}
proc `/`*(lhs: Complex32; rhs: Complex32): Complex32 {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
proc `/`*(lhs: Complex64; rhs: Complex64): Complex64 {.cdecl, importcpp: "(# / #)",
    header : "arrayfire.h".}
proc `+`*(lhs: Complex32; rhs: Complex64): Complex32 {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}
#proc `+`*(lhs: Complex64; rhs: Complex64): Complex64 {.cdecl, importcpp: "(# + #)",header : "arrayfire.h".}
proc `/`*(lhs: Complex32; rhs: Complex64): Complex32 {.cdecl, importcpp: "(# / #)", header : "arrayfire.h".}
#proc `/`*(lhs: Complex64; rhs: Complex64): Complex64 {.cdecl, importcpp: "(# / #)",header : "arrayfire.h".}
proc `==`*(lhs: Complex32; rhs: Complex32): bool {.cdecl, importcpp: "(# == #)", header : "arrayfire.h".}
proc `==`*(lhs: Complex64; rhs: Complex64): bool {.cdecl, importcpp: "(# == #)",header : "arrayfire.h".}
proc abs*(val: Complex32): cfloat {.cdecl, importcpp: "abs(@)", header : "arrayfire.h".}
proc abs*(val: Complex64): cdouble {.cdecl, importcpp: "abs(@)", header : "arrayfire.h".}
proc conj*(val: Complex32): cfloat {.cdecl, importcpp: "conj(@)", header : "arrayfire.h".}
proc conj*(val: Complex64): cdouble {.cdecl, importcpp: "conj(@)", header : "arrayfire.h".}

proc constant*[T](val: T; dims: Dim4; ty: Dtype ): Matrix 
  {.cdecl,importcpp: "af::constant(@)", header : "arrayfire.h".}

proc constant*[T](val: T; dims: Dim4): Matrix 
  {.cdecl,importcpp: "af::constant(@)", header : "arrayfire.h".}

proc constant*[T](val: T; d0: DimT; ty: Dtype ): Matrix 
  {.cdecl,importcpp: "af::constant(@)", header : "arrayfire.h".}

proc constant*[T](val: T; d0: DimT): Matrix 
  {.cdecl,importcpp: "af::constant(@)", header : "arrayfire.h".}

proc constant*[T](val: T; d0: DimT; d1: DimT; ty: Dtype): Matrix 
  {.cdecl, importcpp: "af::constant(@)", header : "arrayfire.h".}

proc constant*[T](val: T; d0: DimT; d1: DimT; d2: DimT; ty: Dtype): Matrix 
  {.cdecl, importcpp: "af::constant(@)", header : "arrayfire.h".}

proc constant*[T](val: T; d0: DimT; d1: DimT; d2: DimT): Matrix 
  {.cdecl, importcpp: "af::constant(@)", header : "arrayfire.h".}

proc constant*[T](val: T; d0: DimT; d1: DimT; d2: DimT; d3: DimT;ty: Dtype = f64 ): Matrix 
  {.cdecl,importcpp: "af::constant(@)", header : "arrayfire.h".}

proc constant*[T](val: T; d0: DimT; d1: DimT; d2: DimT; d3: DimT): Matrix 
  {.cdecl,importcpp: "af::constant(@)", header : "arrayfire.h".}

proc identity*(dims: Dim4; ty: Dtype = f32): Matrix {.cdecl, importcpp: "af::identity(@)",
    header : "arrayfire.h".}

proc identity*(d0: DimT; ty: Dtype = f32): Matrix {.cdecl, importcpp: "af::identity(@)",
    header : "arrayfire.h".}

proc identity*(d0: DimT; d1: DimT; ty: Dtype = f32): Matrix {.cdecl,
    importcpp: "af::identity(@)", header : "arrayfire.h".}

proc identity*(d0: DimT; d1: DimT; d2: DimT; ty: Dtype = f32): Matrix {.cdecl,
    importcpp: "af::identity(@)", header : "arrayfire.h".}

proc identity*(d0: DimT; d1: DimT; d2: DimT; d3: DimT; ty: Dtype = f32): Matrix {.cdecl,
    importcpp: "af::identity(@)", header : "arrayfire.h".}

proc range*(dims: Dim4; afp_seq_dim: cint = - 1; ty: Dtype = f32): Matrix {.cdecl,
    importcpp: "af::range(@)", header : "arrayfire.h".}

proc range*(d0: DimT; d1: DimT = 1; d2: DimT = 1; d3: DimT = 1; afp_seq_dim: cint = - 1;
           ty: Dtype = f32): Matrix 
    {.cdecl, importcpp: "af::range(@)", header : "arrayfire.h".}

proc iota*(dims: Dim4; tileDims: Dim4 ; ty: Dtype ): Matrix {.cdecl,
    importcpp: "iota(@)", header : "arrayfire.h".}

proc diag*(matin : Matrix; num: cint = 0; extract: bool = true): Matrix {.cdecl,
    importcpp: "af::diag(@)", header : "arrayfire.h".}

proc join*(dim: cint; first: Matrix; second: Matrix): Matrix {.cdecl,
    importcpp: "af::join(@)", header : "arrayfire.h".}

proc join*(dim: cint; first: Matrix; second: Matrix; third: Matrix): Matrix {.cdecl,
    importcpp: "af::join(@)", header : "arrayfire.h".}

proc join*(dim: cint; first: Matrix; second: Matrix; third: Matrix; fourth: Matrix): Matrix {.
    cdecl, importcpp: "af::join(@)", header : "arrayfire.h".}

proc tile*(matin : Matrix; x: cuint; y: cuint = 1; z: cuint = 1; w: cuint = 1): Matrix 
  {.cdecl, importcpp: "af::tile(@)", header : "arrayfire.h".}

proc tile*(matin : Matrix; dims: Dim4): Matrix 
  {.cdecl, importcpp: "af::tile(@)", header : "arrayfire.h".}

proc reorder*(matin : Matrix; x: cuint; y: cuint = 1; z: cuint = 2; w: cuint = 3): Matrix 
  {.cdecl, importcpp: "af::reorder(@)", header : "arrayfire.h".}

proc shift*(matin : Matrix; x: cint; y: cint = 0; z: cint = 0; w: cint = 0): Matrix 
  {.cdecl, importcpp: "af::shift(@)", header : "arrayfire.h".}

proc moddims*(matin : Matrix; ndims: cuint; dims: ptr DimT): Matrix 
  {.cdecl, importcpp: "af::moddims(@)", header : "arrayfire.h".}

proc moddims*(matin : Matrix; dims: Dim4): Matrix 
  {.cdecl, importcpp: "af::moddims(@)", header : "arrayfire.h".}

proc moddims*(matin : Matrix; d0: DimT; d1: DimT = 1; d2: DimT = 1; d3: DimT = 1): Matrix {.cdecl,
    importcpp: "moddims(@)", header : "arrayfire.h".}

proc flat*(matin : Matrix): Matrix 
  {.cdecl, importcpp: "af::flat(@)", header : "arrayfire.h".}

proc flip*(matin : Matrix; dim: cuint): Matrix 
  {.cdecl, importcpp: "af::flip(@)", header : "arrayfire.h".}

proc lower*(matin : Matrix; isUnitDiag: bool = false): Matrix 
  {.cdecl, importcpp: "af::lower(@)", header : "arrayfire.h".}

proc upper*(matin : Matrix; isUnitDiag: bool = false): Matrix 
  {.cdecl, importcpp: "af::upper(@)", header : "arrayfire.h".}

proc select*(cond: Matrix; a: Matrix; b: Matrix): Matrix 
  {.cdecl, importcpp: "af::select(@)", header : "arrayfire.h".}

proc select*(cond: Matrix; a: Matrix; b: cdouble): Matrix 
  {.cdecl, importcpp: "af::select(@)", header : "arrayfire.h".}

proc select*(cond: Matrix; a: cdouble; b: Matrix): Matrix 
  {.cdecl, importcpp: "af::select(@)", header : "arrayfire.h".}

proc replace*(a: var Matrix; cond: Matrix; b: Matrix) 
  {.cdecl, importcpp: "af::replace(@)", header : "arrayfire.h".}

proc replace*(a: var Matrix; cond: Matrix; b: cdouble) 
  {.cdecl, importcpp: "af::replace(@)", header : "arrayfire.h".}

proc info*() 
  {.cdecl, importcpp: "af::info(@)", header : "arrayfire.h".}

proc infoString*(verbose: bool = false): cstring 
  {.cdecl, importcpp: "af::infoString(@)", header : "arrayfire.h".}

proc deviceInfo*(dName: cstring; dPlatform: cstring; dToolkit: cstring; dCompute: cstring) 
  {.cdecl, importcpp: "af::deviceInfo(@)", header : "arrayfire.h".}

proc getDeviceCount*(): cint 
  {.cdecl, importcpp: "af::getDeviceCount(@)", header : "arrayfire.h".}

proc getDevice*(): cint 
  {.cdecl, importcpp: "af::getDevice(@)", header : "arrayfire.h".}

proc isDoubleAvailable*(device: cint): bool {.cdecl,
    importcpp: "af::isDoubleAvailable(@)", header : "arrayfire.h".}

proc isHalfAvailable*(device: cint): bool {.cdecl,
    importcpp: "af::isHalfAvailable(@)", header : "arrayfire.h".}

proc setDevice*(device: cint) 
  {.cdecl, importcpp: "af::setDevice(@)", header : "arrayfire.h".}

proc sync*(device: cint = - 1) 
  {.cdecl, importcpp: "af::sync(@)", header : "arrayfire.h".}

proc af_alloc*(elements: csize_t; `type`: Dtype): pointer 
  {.cdecl, importcpp: "af::alloc(@)",header : "arrayfire.h".}

proc af_alloc*[T](elements: csize_t): ptr T 
  {.cdecl, importcpp: "af::alloc(@)", header : "arrayfire.h".}

proc af_free*(`ptr`: pointer) 
  {.cdecl, importcpp: "af::free(@)", header : "arrayfire.h".}

proc pinned*(elements: csize_t; `type`: Dtype): pointer 
  {.cdecl, importcpp: "af::pinned(@)", header : "arrayfire.h".}

proc pinned*[T](elements: csize_t): ptr T 
  {.cdecl, importcpp: "af::pinned(@)", header : "arrayfire.h".}

proc freePinned*(`ptr`: pointer) 
  {.cdecl, importcpp: "af::freePinned(@)", header : "arrayfire.h".}

proc allocHost*(elements: csize_t; `type`: Dtype): pointer 
  {.cdecl, importcpp: "af::allocHost(@)", header : "arrayfire.h".}

proc allocHost*[T](elements: csize_t): ptr T 
  {.cdecl, importcpp: "af::allocHost(@)", header : "arrayfire.h".}

proc freeHost*(`ptr`: pointer) 
  {.cdecl, importcpp: "af::freeHost(@)", header : "arrayfire.h".}

proc deviceMemInfo*(allocBytes: ptr csize_t; allocBuffers: ptr csize_t;
                   lockBytes: ptr csize_t; lockBuffers: ptr csize_t) 
  {.cdecl, importcpp: "af::deviceMemInfo(@)", header : "arrayfire.h".}

proc printMemInfo*(msg: cstring = nil; deviceId: cint = - 1) 
  {.cdecl, importcpp: "af::printMemInfo(@)", header : "arrayfire.h".}

proc deviceGC*() 
  {.cdecl, importcpp: "af::deviceGC(@)", header : "arrayfire.h".}

proc setMemStepSize*(size: csize_t) 
  {.cdecl, importcpp: "af::setMemStepSize(@)", header : "arrayfire.h".}

proc getMemStepSize*(): csize_t 
  {.cdecl, importcpp: "af::getMemStepSize(@)", header : "arrayfire.h".}

proc constructfeatures*(): Features 
  {.cdecl, constructor, importcpp: "af::features(@)",header : "arrayfire.h".}

proc constructfeatures*(n: csize_t): Features 
  {.cdecl, constructor, importcpp: "af::features(@)", header : "arrayfire.h".}

proc constructfeatures*(f: Features): Features 
  {.cdecl, constructor, importcpp: "af::features(@)", header : "arrayfire.h".}

proc destroyfeatures*(this: var Features) 
  {.cdecl, importcpp: "#.~features()", header : "arrayfire.h".}

proc getNumFeatures*(this: Features): csize_t 
  {.noSideEffect, cdecl, importcpp: "getNumFeatures", header : "arrayfire.h".}

proc getX*(this: Features): Matrix 
  {.noSideEffect, cdecl, importcpp: "getX", header : "arrayfire.h".}

proc getY*(this: Features): Matrix 
  {.noSideEffect, cdecl, importcpp: "getY", header : "arrayfire.h".}

proc getScore*(this: Features): Matrix 
  {.noSideEffect, cdecl, importcpp: "getScore", header : "arrayfire.h".}

proc getOrientation*(this: Features): Matrix 
  {.noSideEffect, cdecl, importcpp: "getOrientation", header : "arrayfire.h".}

proc getSize*(this: Features): Matrix 
  {.noSideEffect, cdecl, importcpp: "getSize", header : "arrayfire.h".}

proc get*(this: Features): AFC_Features 
  {.noSideEffect, cdecl, importcpp: "get", header : "arrayfire.h".}

proc gforToggle*(): bool 
  {.cdecl, importcpp: "af::gforToggle(@)", header : "arrayfire.h".}

proc gforGet*(): bool 
  {.cdecl, importcpp: "af::gforGet(@)", header : "arrayfire.h".}

proc gforSet*(val: bool) 
  {.cdecl, importcpp: "af::gforSet(@)", header : "arrayfire.h".}

proc batchFunc*(lhs: Matrix; rhs: Matrix; `func`: BatchFuncT): Matrix {.cdecl,
    importcpp: "af::batchFunc(@)", header : "arrayfire.h".}

proc afwindow*(): Window 
  {.cdecl, constructor, importcpp: "af::Window(@)", header : "arrayfire.h".}

proc afwindow*(title: cstring): Window 
  {.cdecl, constructor, importcpp: "af::Window(@)", header : "arrayfire.h".}

proc afwindow*(width: cint; height: cint; title: cstring = "ArrayFire"): Window 
  {.cdecl, constructor, importcpp: "af::Window(@)", header : "arrayfire.h".}

proc afwindow*(wnd: Window): Window 
  {.cdecl, constructor, importcpp: "af::Window(@)", header : "arrayfire.h".}

proc destroyWindow*(this: var Window) 
  {.cdecl, importcpp: "#.~Window()", header : "arrayfire.h".}

proc get*(this: Window): Window 
  {.noSideEffect, cdecl, importcpp: "get", header : "arrayfire.h".}

proc setPos*(this: var Window; x: cuint; y: cuint) 
  {.cdecl, importcpp: "setPos",header : "arrayfire.h".}

proc setTitle*(this: var Window; title: cstring) 
  {.cdecl, importcpp: "setTitle", header : "arrayfire.h".}

proc setSize*(this: var Window; w: cuint; h: cuint) 
  {.cdecl, importcpp: "setSize", header : "arrayfire.h".}

proc setColorMap*(this: var Window; cmap: ColorMap) 
  {.cdecl, importcpp: "setColorMap", header : "arrayfire.h".}

proc image*(this: var Window; matin : Matrix; title: cstring = nil) 
  {.cdecl, importcpp: "image", header : "arrayfire.h".}

proc plot*(this: var Window; matin : Matrix; title: cstring = nil) 
  {.cdecl, importcpp: "plot", header : "arrayfire.h".}

proc plot*(this: var Window; x: Matrix; y: Matrix; z: Matrix; title: cstring = nil) 
  {.cdecl, importcpp: "plot", header : "arrayfire.h".}

proc plot*(this: var Window; x: Matrix; y: Matrix; title: cstring = nil) 
  {.cdecl, importcpp: "plot", header : "arrayfire.h".}

proc scatter*(this: var Window; matin : Matrix; marker: MarkerType ; title: cstring = nil) 
  {.cdecl, importcpp: "scatter", header : "arrayfire.h".}

proc scatter*(this: var Window; x: Matrix; y: Matrix; z: Matrix; marker: MarkerType ; title: cstring ) 
  {.cdecl, importcpp: "scatter", header : "arrayfire.h".}

proc scatter*(this: var Window; x: Matrix; y: Matrix; marker: MarkerType ; title: cstring = nil) 
  {.cdecl, importcpp: "scatter", header : "arrayfire.h".}

proc hist*(this: var Window; x: Matrix; minval: cdouble; maxval: cdouble; title: cstring = nil) 
  {.cdecl, importcpp: "hist", header : "arrayfire.h".}

proc surface*(this: var Window; s: Matrix; title: cstring = nil) 
  {.cdecl, importcpp: "surface", header : "arrayfire.h".}

proc surface*(this: var Window; xVals: Matrix; yVals: Matrix; s: Matrix; title: cstring = nil) 
  {.cdecl, importcpp: "surface", header : "arrayfire.h".}

proc vectorField*(this: var Window; points: Matrix; directions: Matrix; title: cstring = nil) 
  {.cdecl, importcpp: "vectorField", header : "arrayfire.h".}

proc vectorField*(this: var Window; xPoints: Matrix; yPoints: Matrix; zPoints: Matrix;
                 xDirs: Matrix; yDirs: Matrix; zDirs: Matrix; title: cstring = nil) 
  {.cdecl, importcpp: "vectorField", header : "arrayfire.h".}

proc vectorField*(this: var Window; xPoints: Matrix; yPoints: Matrix; xDirs: Matrix;
                 yDirs: Matrix; title: cstring = nil) 
  {.cdecl, importcpp: "vectorField", header : "arrayfire.h".}

proc setAxesLimits*(this: var Window; x: Matrix; y: Matrix; exact: bool = false) 
  {.cdecl, importcpp: "setAxesLimits", header : "arrayfire.h".}

proc setAxesLimits*(this: var Window; x: Matrix; y: Matrix; z: Matrix; exact: bool = false) 
  {.cdecl, importcpp: "setAxesLimits", header : "arrayfire.h".}

proc setAxesLimits*(this: var Window; xmin: cfloat; xmax: cfloat; ymin: cfloat;
                   ymax: cfloat; exact: bool = false) 
  {.cdecl,importcpp: "setAxesLimits", header : "arrayfire.h".}

proc setAxesLimits*(this: var Window; xmin: cfloat; xmax: cfloat; ymin: cfloat;
                   ymax: cfloat; zmin: cfloat; zmax: cfloat; exact: bool = false) 
  {.cdecl, importcpp: "setAxesLimits", header : "arrayfire.h".}

proc setAxesTitles*(this: var Window; xtitle: cstring = "X-Axis";
                   ytitle: cstring = "Y-Axis"; ztitle: cstring = "Z-Axis") 
  {.cdecl, importcpp: "setAxesTitles", header : "arrayfire.h".}

proc grid*(this: var Window; rows: cint; cols: cint) 
  {.cdecl, importcpp: "#.grid(@)", header : "arrayfire.h".}

proc show*(this: var Window) 
  {.cdecl, importcpp: "show", header : "arrayfire.h".}

proc close*(this: var Window): bool 
  {.cdecl, importcpp: "close", header : "arrayfire.h".}

proc setVisibility*(this: var Window; isVisible: bool) 
  {.cdecl, importcpp: "setVisibility", header : "arrayfire.h".}

proc `[]`*(this: var Window; r: cint; c: cint): var Window 
  {.cdecl, importcpp: "#(@)", header : "arrayfire.h".}

proc grad*(dx: var Matrix; dy: var Matrix; matin : Matrix) 
  {.cdecl, importcpp: "af::grad(@)", header : "arrayfire.h".}

proc grad*(matin: Matrix) : tuple[dx: Matrix, dy : Matrix] =
  var dx : Matrix
  var dy : Matrix 

  grad(dx, dy, matin)
  (dx, dy)

proc loadImage*(filename: cstring; isColor: bool = false): Matrix 
  {.cdecl, importcpp: "af::loadImage(@)", header : "arrayfire.h".}

proc saveImage*(filename: cstring; matin : Matrix) 
  {.cdecl, importcpp: "af::saveImage(@)", header : "arrayfire.h".}

proc loadImageMem*(`ptr`: pointer): Matrix 
  {.cdecl, importcpp: "af::loadImageMem(@)", header : "arrayfire.h".}

proc saveImageMem*(matin : Matrix; format: ImageFormat ): pointer 
  {.cdecl, importcpp: "af::saveImageMem(@)", header : "arrayfire.h".}

proc deleteImageMem*(`ptr`: pointer) 
  {.cdecl, importcpp: "af::deleteImageMem(@)", header : "arrayfire.h".}

proc loadImageNative*(filename: cstring): Matrix 
  {.cdecl, importcpp: "af::loadImageNative(@)", header : "arrayfire.h".}

proc saveImageNative*(filename: cstring; matin : Matrix) 
  {.cdecl, importcpp: "af::saveImageNative(@)", header : "arrayfire.h".}

proc isImageIOAvailable*(): bool 
  {.cdecl, importcpp: "af::isImageIOAvailable(@)", header : "arrayfire.h".}

proc resize*(matin : Matrix; odim0: DimT; odim1: DimT;
            `method`: InterpType ): Matrix 
  {.cdecl, importcpp: "af::resize(@)", header : "arrayfire.h".}

proc resize*(scale0: cfloat; scale1: cfloat; matin : Matrix;
            `method`: InterpType ): Matrix 
  {.cdecl, importcpp: "af::resize(@)", header : "arrayfire.h".}

proc resize*(scale: cfloat; matin : Matrix; `method`: InterpType ): Matrix 
  {.cdecl, importcpp: "af::resize(@)", header : "arrayfire.h".}

proc rotate*(matin : Matrix; theta: cfloat; crop: bool = true;
            `method`: InterpType ): Matrix 
  {.cdecl, importcpp: "af::rotate(@)", header : "arrayfire.h".}

proc transform*(matin : Matrix; transform: Matrix; odim0: DimT = 0; odim1: DimT = 0;
               `method`: InterpType ; inverse: bool = true): Matrix 
  {.cdecl, importcpp: "af::transform(@)", header : "arrayfire.h".}

proc transformCoordinates*(tf: Matrix; d0: cfloat; d1: cfloat): Matrix 
  {.cdecl, importcpp: "af::transformCoordinates(@)", header : "arrayfire.h".}

proc translate*(matin : Matrix; trans0: cfloat; trans1: cfloat; odim0: DimT = 0;
               odim1: DimT = 0; `method`: InterpType): Matrix 
  {.cdecl, importcpp: "af::translate(@)", header : "arrayfire.h".}

proc scale*(matin : Matrix; scale0: cfloat; scale1: cfloat; odim0: DimT = 0; odim1: DimT = 0;
           `method`: InterpType): Matrix {.cdecl,
    importcpp: "af::scale(@)", header : "arrayfire.h".}

proc skew*(matin : Matrix; skew0: cfloat; skew1: cfloat; odim0: DimT = 0; odim1: DimT = 0;
          inverse: bool = true; `method`: InterpType ): Matrix 
  {.cdecl, importcpp: "af::skew(@)", header : "arrayfire.h".}

proc bilateral*(matin : Matrix; spatialSigma: cfloat; chromaticSigma: cfloat;
                isColor: bool = false): Matrix 
  {.cdecl, importcpp: "af::bilateral(@)", header : "arrayfire.h".}

proc anisotropicDiffusion*(matin : Matrix; timestep: cfloat; conductance: cfloat; iterations: cuint; 
                          fftype: FluxFuction = FluxFuction.EXPONENTIAL; diffusionKind: DiffusionEq = DiffusionEq.GRAD): Matrix 
  {.cdecl, importcpp: "af::anisotropicDiffusion(@)", header : "arrayfire.h".}

proc inverseDeconv*(matin : Matrix; psf: Matrix, gamme: cfloat, algo: InverseDeconvAlgo): Matrix 
  {.cdecl, importcpp: "af::cainverseDeconvnny(@)", header : "arrayfire.h".}

proc iterativeDeconv*(matin : Matrix; ker: Matrix, iterations: cuint, relaxFactor: cfloat, algo: InverseDeconvAlgo): Matrix 
  {.cdecl, importcpp: "af::iterativeDeconv(@)", header : "arrayfire.h".}

proc canny*(matin : Matrix; thresholdType: CannyThreshold; lowThresholdRatio: cfloat; highThresholdRatio: cfloat; sobelWindow: cuint, isFast: bool = false;
            isColor: bool = false): Matrix 
  {.cdecl, importcpp: "af::canny(@)", header : "arrayfire.h".}

proc confidenceCC*(matin : Matrix; seeds: Matrix, radius: cuint, multiplier: cuint, iter: cint, segmentedValue: cdouble): Matrix 
  {.cdecl, importcpp: "af::confidenceCC(@)", header : "arrayfire.h".}

proc histogram*(matin : Matrix; nbins: cuint; minval: cdouble; maxval: cdouble): Matrix 
  {.cdecl, importcpp: "af::histogram(@)", header : "arrayfire.h".}

proc histogram*(matin : Matrix; nbins: cuint): Matrix 
  {.cdecl, importcpp: "af::histogram(@)",header : "arrayfire.h".}

proc meanShift*(matin : Matrix; spatialSigma: cfloat; chromaticSigma: cfloat;
               iter: cuint; isColor: bool = false): Matrix 
  {.cdecl, importcpp: "af::meanShift(@)", header : "arrayfire.h".}

proc minfilt*(matin : Matrix; windLength: DimT = 3; windWidth: DimT = 3;
             edgePad: BorderType ): Matrix 
  {.cdecl, importcpp: "af::minfilt(@)", header : "arrayfire.h".}

proc maxfilt*(matin : Matrix; windLength: DimT = 3; windWidth: DimT = 3;
             edgePad: BorderType ): Matrix 
  {.cdecl, importcpp: "af::maxfilt(@)", header : "arrayfire.h".}

proc dilate*(matin : Matrix; mask: Matrix): Matrix 
  {.cdecl, importcpp: "af::dilate(@)",header : "arrayfire.h".}

proc dilate3*(matin : Matrix; mask: Matrix): Matrix 
  {.cdecl, importcpp: "af::dilate3(@)", header : "arrayfire.h".}

proc erode*(matin : Matrix; mask: Matrix): Matrix 
  {.cdecl, importcpp: "af::erode(@)", header : "arrayfire.h".}

proc erode3*(matin : Matrix; mask: Matrix): Matrix 
  {.cdecl, importcpp: "af::erode3(@)", header : "arrayfire.h".}

proc regions*(matin : Matrix; connectivity: Connectivity;
             `type`: Dtype = f32): Matrix 
  {.cdecl, importcpp: "af::regions(@)", header : "arrayfire.h".}

proc sobel*(dx: var Matrix; dy: var Matrix; img: Matrix; kerSize: cuint = 3) 
  {.cdecl, importcpp: "af::sobel(@)", header : "arrayfire.h".}

proc sobel*(img: Matrix; kerSize: cuint = 3; isFast: bool = false): Matrix 
  {.cdecl, importcpp: "af::sobel(@)", header : "arrayfire.h".}

proc rgb2gray*(matin : Matrix; rPercent: cfloat = 0.2126;
              gPercent: cfloat = 0.7151999999999999; bPercent: cfloat = 0.0722): Matrix 
  {.cdecl, importcpp: "af::rgb2gray(@)", header : "arrayfire.h".}

proc gray2rgb*(matin : Matrix; rFactor: cfloat = 1.0; gFactor: cfloat = 1.0;
              bFactor: cfloat = 1.0): Matrix 
  {.cdecl, importcpp: "af::gray2rgb(@)", header : "arrayfire.h".}

proc histEqual*(matin : Matrix; hist: Matrix): Matrix 
  {.cdecl, importcpp: "af::histEqual(@)",header : "arrayfire.h".}

proc gaussianKernel*(rows: cint; cols: cint; sigR: cdouble = 0; sigC: cdouble = 0): Matrix 
  {.cdecl, importcpp: "af::gaussianKernel(@)", header : "arrayfire.h".}

proc hsv2rgb*(matin : Matrix): Matrix 
  {.cdecl, importcpp: "af::hsv2rgb(@)", header : "arrayfire.h".}

proc rgb2hsv*(matin : Matrix): Matrix 
  {.cdecl, importcpp: "af::rgb2hsv(@)", header : "arrayfire.h".}

proc colorSpace*(image: Matrix; to: CSpace; `from`: CSpace): Matrix 
  {.cdecl, importcpp: "af::colorSpace(@)", header : "arrayfire.h".}

proc unwrap*(matin : Matrix; wx: DimT; wy: DimT; sx: DimT; sy: DimT; px: DimT = 0; py: DimT = 0;
            isColumn: bool = true): Matrix 
  {.cdecl, importcpp: "af::unwrap(@)", header : "arrayfire.h".}

proc wrap*(matin : Matrix; ox: DimT; oy: DimT; wx: DimT; wy: DimT; sx: DimT; sy: DimT;
          px: DimT = 0; py: DimT = 0; isColumn: bool = true): Matrix 
  {.cdecl, importcpp: "af::wrap(@)", header : "arrayfire.h".}

proc sat*(matin : Matrix): Matrix 
  {.cdecl, importcpp: "af::sat(@)", header : "arrayfire.h".}

proc ycbcr2rgb*(matin : Matrix; standard: YCCStd ): Matrix 
  {.cdecl, importcpp: "af::ycbcr2rgb(@)", header : "arrayfire.h".}

proc rgb2ycbcr*(matin : Matrix; standard: YCCStd ): Matrix 
  {.cdecl, importcpp: "af::rgb2ycbcr(@)", header : "arrayfire.h".}

proc moments*(`out`: ptr cdouble; matin : Matrix;
             moment: MomentType ) 
  {.cdecl,importcpp: "af::moments(@)", header : "arrayfire.h".}

proc moments*(matin : Matrix; moment: MomentType ): Matrix 
  {.cdecl, importcpp: "af::moments(@)", header : "arrayfire.h".}

proc af_svd*(u: var Matrix; s: var Matrix; vt: var Matrix; matin : Matrix) 
  {.cdecl, importcpp: "af::svd(@)", header : "arrayfire.h".}

proc svd*(matin: Matrix) : tuple[u: Matrix, s: Matrix, vt: Matrix] =
  var u : Matrix
  var s : Matrix
  var v : Matrix
  af_svd(u, s, v, matin)
  (u,s,v)


proc svdInPlace*(u: var Matrix; s: var Matrix; vt: var Matrix; matin: var Matrix) 
  {.cdecl, importcpp: "af::svdInPlace(@)", header : "arrayfire.h".}

proc lu*(matout: var Matrix; pivot: var Matrix; matin : Matrix; isLapackPiv: bool = true) 
  {.cdecl, importcpp: "af::lu(@)", header : "arrayfire.h".}

proc lu*(lower: var Matrix; upper: var Matrix; pivot: var Matrix; matin : Matrix) 
  {.cdecl, importcpp: "af::lu(@)", header : "arrayfire.h".}

proc luInPlace*(pivot: var Matrix; `in`: var Matrix; isLapackPiv: bool = true) 
  {.cdecl, importcpp: "af::luInPlace(@)", header : "arrayfire.h".}

proc qr*(`out`: var Matrix; tau: var Matrix; matin : Matrix) 
  {.cdecl, importcpp: "af::qr(@)", header : "arrayfire.h".}

proc qr*(q: var Matrix; r: var Matrix; tau: var Matrix; matin : Matrix) 
  {.cdecl, importcpp: "af::qr(@)", header : "arrayfire.h".}

proc qrInPlace*(tau: var Matrix; `in`: var Matrix) 
  {.cdecl, importcpp: "af::qrInPlace(@)", header : "arrayfire.h".}

proc cholesky*(`out`: var Matrix; matin : Matrix; isUpper: bool = true): cint 
  {.cdecl, importcpp: "af::cholesky(@)", header : "arrayfire.h".}

proc choleskyInPlace*(`in`: var Matrix; isUpper: bool = true): cint 
  {.cdecl, importcpp: "af::choleskyInPlace(@)", header : "arrayfire.h".}

proc solve*(a: Matrix; b: Matrix; options: MatProp ): Matrix 
  {.cdecl, importcpp: "af::solve(@)", header : "arrayfire.h".}

proc solveLU*(a: Matrix; piv: Matrix; b: Matrix; options: MatProp ): Matrix 
  {.cdecl, importcpp: "af::solveLU(@)", header : "arrayfire.h".}

proc inverse*(matin : Matrix; options: MatProp ): Matrix 
  {.cdecl, importcpp: "af::inverse(@)", header : "arrayfire.h".}

proc pinverse*(matin : Matrix; options: MatProp ): Matrix 
  {.cdecl, importcpp: "af::pinverse(@)", header : "arrayfire.h".}

proc rank*(matin : Matrix; tol: cdouble = 1e-06, options: MatProp = MatProp.NONE): Matrix
  {.cdecl, importcpp: "af::rank(@)", header : "arrayfire.h".}

proc det*(matin : Matrix): cdouble
  {.cdecl, importcpp: "af::det<double>(@)", header : "arrayfire.h".}

proc norm*(matin : Matrix; `type`: NormType ; p: cdouble = 1; q: cdouble = 1): cdouble 
  {.cdecl, importcpp: "af::norm(@)", header : "arrayfire.h".}

proc isLAPACKAvailable*(): bool 
  {.cdecl, importcpp: "af::isLAPACKAvailable(@)", header : "arrayfire.h".}

proc constructrandomEngine*(typeIn: RandomEngineType; seedIn: uintl = 0): RandomEngine 
  {.cdecl, constructor, importcpp: "af::randomEngine(@)", header : "arrayfire.h".}

proc constructrandomEngine*(`in`: RandomEngine): RandomEngine 
  {.cdecl, constructor, importcpp: "af::randomEngine(@)", header : "arrayfire.h".}

proc constructrandomEngine*(engine: RandomEngine): RandomEngine 
  {.cdecl, constructor, importcpp: "af::randomEngine(@)", header : "arrayfire.h".}

proc destroyrandomEngine*(this: var RandomEngine) 
  {.cdecl, importcpp: "#.~randomEngine()", header : "arrayfire.h".}

proc setType*(this: var RandomEngine; `type`: RandomEngineType) 
  {.cdecl, importcpp: "setType", header : "arrayfire.h".}

proc getType*(this: var RandomEngine): RandomEngineType 
  {.cdecl, importcpp: "getType", header : "arrayfire.h".}

proc setSeed*(this: var RandomEngine; seed: uintl) 
  {.cdecl, importcpp: "setSeed(@)", header : "arrayfire.h".}

proc getSeed*(this: RandomEngine): uintl 
  {.noSideEffect, cdecl, importcpp: "getSeed(@)", header : "arrayfire.h".}

proc setSeed*(seed: uintl) 
  {.cdecl, importcpp: "af::setSeed(@)", header : "arrayfire.h".}

proc getSeed*(): uintl 
  {.noSideEffect, cdecl, importcpp: "af::getSeed(@)", header : "arrayfire.h".}


proc get*(this: RandomEngine): AFC_RandomEngine 
  {.noSideEffect, cdecl, importcpp: "get", header : "arrayfire.h".}

proc setDefaultRandomEngineType*(rtype: RandomEngineType) 
  {.cdecl, importcpp: "af::setDefaultRandomEngineType(@)", header : "arrayfire.h".}

proc getDefaultRandomEngine*(): RandomEngine 
  {.cdecl, importcpp: "af::getDefaultRandomEngine(@)", header : "arrayfire.h".}

proc randu*(dims: Dim4; ty: Dtype; r: var RandomEngine ): Matrix 
  {.cdecl, importcpp: "af::randu(@)", header : "arrayfire.h".}

proc randn*(dims: Dim4; ty: Dtype; r: var RandomEngine): Matrix 
  {.cdecl, importcpp: "randn(@)", header : "arrayfire.h".}

proc randu*(dims: Dim4; ty: Dtype = f32): Matrix 
  {.cdecl, importcpp: "af::randu(@)", header : "arrayfire.h".}

proc randu*(d0: DimT; ty: Dtype = f32): Matrix 
  {.cdecl, importcpp: "af::randu(@)", header : "arrayfire.h".}

proc randu*(d0, d1: DimT; ty: Dtype = f32): Matrix 
  {.cdecl, importcpp: "af::randu(@)", header : "arrayfire.h".}

proc randu*(d0, d1, d2: DimT; ty: Dtype = f32): Matrix 
  {.cdecl, importcpp: "af::randu(@)", header : "arrayfire.h".}

proc randu*(d0, d1, d2, d3: DimT; ty: Dtype = f32): Matrix 
  {.cdecl, importcpp: "af::randu(@)", header : "arrayfire.h".}

proc randn*(dims: Dim4; ty: Dtype = f32): Matrix 
  {.cdecl, importcpp: "af::randn(@)", header : "arrayfire.h".}

proc randn*(d0: DimT; ty: Dtype = f32): Matrix 
  {.cdecl, importcpp: "af::randn(@)", header : "arrayfire.h".}

proc randn*(d0, d1: DimT; ty: Dtype = f32): Matrix 
  {.cdecl, importcpp: "af::randn(@)", header : "arrayfire.h".}

proc randn*(d0, d1, d2: DimT; ty: Dtype = f32): Matrix 
  {.cdecl, importcpp: "af::randn(@)", header : "arrayfire.h".}

proc randn*(d0, d1, d2, d3: DimT; ty: Dtype = f32): Matrix 
  {.cdecl, importcpp: "af::randn(@)", header : "arrayfire.h".}

proc mseq*(length: cdouble = 0): AF_Seq 
  {.cdecl, constructor, importcpp: "af::seq(@)", header : "arrayfire.h".}

proc destroyseq*(this: var AF_Seq)  {.cdecl, importcpp: "#.~seq()", header : "arrayfire.h".}

proc mseq*(begin: cdouble; last: cdouble; step: cdouble = 1): AF_Seq 
  {.cdecl, constructor, importcpp: "af::seq(@)", header : "arrayfire.h".}

proc mseq*(afs: AF_SEQ; isGfor: bool): AF_Seq 
  {.cdecl, constructor,  importcpp: "af::seq(@)", header : "arrayfire.h".}

proc mseq*(s: AF_Seq): AF_Seq
  {.cdecl, constructor, importcpp: "seq(@)", header : "arrayfire.h".}

proc `-`*(this: var AF_SEQ): AF_SEQ 
  {.cdecl, importcpp: "(- #)", header : "arrayfire.h".}

proc `+`*(this: var AF_SEQ; x: cdouble): AF_SEQ 
  {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}

proc `-`*(this: var AF_SEQ; x: cdouble): AF_SEQ 
  {.cdecl, importcpp: "(# - #)",
    header : "arrayfire.h".}
proc `*`*(this: var AF_SEQ; x: cdouble): AF_SEQ 
  {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}

proc `+`*(this: var AF_SEQ; x: cdouble; y: AF_SEQ): AF_SEQ 
  {.cdecl, importcpp: "(# + #)", header : "arrayfire.h".}

proc `-`*(this: var AF_SEQ; x: cdouble; y: AF_SEQ): AF_SEQ 
  {.cdecl, importcpp: "(# - #)", header : "arrayfire.h".}

proc `*`*(this: var AF_SEQ; x: cdouble; y: AF_SEQ): AF_SEQ 
  {.cdecl, importcpp: "(# * #)", header : "arrayfire.h".}

proc get_Matrix*(this: AF_SEQ): Matrix 
  {.cdecl, importcpp: "#.operator af::array()",header : "arrayfire.h".}

proc get_Matrix*(this: Matrix_View): Matrix 
  {.cdecl, importcpp: "af::array(#)",header : "arrayfire.h".}

proc approx1*(matin : Matrix; pos: Matrix; imethod: InterpType ;
             offGrid: cfloat = 0.0): Matrix 
  {.cdecl, importcpp: "af::approx1(@)", header : "arrayfire.h".}

proc approx2*(matin : Matrix; pos0: Matrix; pos1: Matrix;
             `method`: InterpType ; offGrid: cfloat = 0.0): Matrix 
  {.cdecl, importcpp: "af::approx2(@)", header : "arrayfire.h".}

proc fftNorm*(matin : Matrix; normFactor: cdouble; odim0: DimT = 0): Matrix 
  {.cdecl, importcpp: "af::fftNorm(@)", header : "arrayfire.h".}

proc fft2Norm*(matin : Matrix; normFactor: cdouble; odim0: DimT = 0; odim1: DimT = 0): Matrix 
  {.cdecl, importcpp: "af::fft2Norm(@)", header : "arrayfire.h".}

proc fft3Norm*(matin : Matrix; normFactor: cdouble; odim0: DimT = 0; odim1: DimT = 0;
              odim2: DimT = 0): Matrix 
  {.cdecl, importcpp: "af::fft3Norm(@)", header : "arrayfire.h".}

proc fftInPlace*(`in`: var Matrix; normFactor: cdouble = 1) 
  {.cdecl, importcpp: "af::fftInPlace(@)", header : "arrayfire.h".}

proc fft2InPlace*(`in`: var Matrix; normFactor: cdouble = 1) 
  {.cdecl, importcpp: "af::fft2InPlace(@)", header : "arrayfire.h".}

proc fft3InPlace*(`in`: var Matrix; normFactor: cdouble = 1) 
  {.cdecl, importcpp: "af::fft3InPlace(@)", header : "arrayfire.h".}

proc fft*(matin : Matrix; odim0: DimT = 0): Matrix 
  {.cdecl, importcpp: "af::fft(@)", header : "arrayfire.h".}

proc fft2*(matin : Matrix; odim0: DimT = 0; odim1: DimT = 0): Matrix 
  {.cdecl, importcpp: "af::fft2(@)", header : "arrayfire.h".}

proc fft3*(matin : Matrix; odim0: DimT = 0; odim1: DimT = 0; odim2: DimT = 0): Matrix 
  {.cdecl, importcpp: "af::fft3(@)", header : "arrayfire.h".}

proc dft*(matin : Matrix; normFactor: cdouble; outDims: Dim4): Matrix 
  {.cdecl, importcpp: "af::dft(@)", header : "arrayfire.h".}

proc dft*(matin : Matrix; outDims: Dim4): Matrix {.cdecl, importcpp: "dft(@)",
    header : "arrayfire.h".}

proc dft*(matin : Matrix): Matrix {.cdecl, importcpp: "dft(@)", header : "arrayfire.h".}

proc ifftNorm*(matin : Matrix; normFactor: cdouble; odim0: DimT = 0): Matrix {.cdecl,
    importcpp: "ifftNorm(@)", header : "arrayfire.h".}

proc ifft2Norm*(matin : Matrix; normFactor: cdouble; odim0: DimT = 0; odim1: DimT = 0): Matrix {.
    cdecl, importcpp: "ifft2Norm(@)", header : "arrayfire.h".}

proc ifft3Norm*(matin : Matrix; normFactor: cdouble; odim0: DimT = 0; odim1: DimT = 0;
               odim2: DimT = 0): Matrix {.cdecl, importcpp: "ifft3Norm(@)",
                                     header : "arrayfire.h".}

proc ifftInPlace*(`in`: var Matrix; normFactor: cdouble = 1) {.cdecl,
    importcpp: "ifftInPlace(@)", header : "arrayfire.h".}

proc ifft2InPlace*(`in`: var Matrix; normFactor: cdouble = 1) {.cdecl,
    importcpp: "ifft2InPlace(@)", header : "arrayfire.h".}

proc ifft3InPlace*(`in`: var Matrix; normFactor: cdouble = 1) {.cdecl,
    importcpp: "ifft3InPlace(@)", header : "arrayfire.h".}

proc ifft*(matin : Matrix; odim0: DimT = 0): Matrix {.cdecl, importcpp: "ifft(@)",
    header : "arrayfire.h".}

proc ifft2*(matin : Matrix; odim0: DimT = 0; odim1: DimT = 0): Matrix {.cdecl,
    importcpp: "ifft2(@)", header : "arrayfire.h".}

proc ifft3*(matin : Matrix; odim0: DimT = 0; odim1: DimT = 0; odim2: DimT = 0): Matrix {.cdecl,
    importcpp: "ifft3(@)", header : "arrayfire.h".}

proc idft*(matin : Matrix; normFactor: cdouble; outDims: Dim4): Matrix {.cdecl,
    importcpp: "idft(@)", header : "arrayfire.h".}

proc idft*(matin : Matrix; outDims: Dim4): Matrix {.cdecl, importcpp: "idft(@)",
    header : "arrayfire.h".}

proc idft*(matin : Matrix): Matrix {.cdecl, importcpp: "idft(@)", header : "arrayfire.h".}

proc fftR2C*[Rank](matin : Matrix; dims: Dim4; normFactor: cdouble = 0): Matrix {.cdecl,
    importcpp: "fftR2C(@)", header : "arrayfire.h".}

proc fftR2C*[Rank](matin : Matrix; normFactor: cdouble = 0): Matrix {.cdecl,
    importcpp: "fftR2C(@)", header : "arrayfire.h".}

proc fftC2R*[Rank](matin : Matrix; isOdd: bool = false; normFactor: cdouble = 0): Matrix {.
    cdecl, importcpp: "fftC2R(@)", header : "arrayfire.h".}

proc convolve*(signal: Matrix; filter: Matrix; mode: ConvMode = ConvMode.DEFAULT ;
              domain: ConvDomain = ConvDomain.AUTO): Matrix {.cdecl,
    importcpp: "convolve(@)", header : "arrayfire.h".}

proc convolve*(colFilter: Matrix; rowFilter: Matrix; signal: Matrix;
              mode: ConvMode = ConvMode.DEFAULT ): Matrix {.cdecl,
    importcpp: "convolve(@)", header : "arrayfire.h".}

proc convolve2GradientNN*(incomming_gradient: Matrix; original_signal: Matrix; original_filter: Matrix; convolved_output: Matrix;
                          stride: Dim4; padding: Dim4; dilation: Dim4; gradType: ConvGradientType){.cdecl,
                          importcpp: "convolve(@)", header : "arrayfire.h".}

proc convolve1*(signal: Matrix; filter: Matrix; mode: ConvMode = ConvMode.DEFAULT;
               domain: ConvDomain = ConvDomain.AUTO ): Matrix {.cdecl,
    importcpp: "convolve1(@)", header : "arrayfire.h".}

proc convolve2*(signal: Matrix; filter: Matrix; mode: ConvMode = ConvMode.DEFAULT ;
               domain: ConvDomain = ConvDomain.AUTO ): Matrix {.cdecl,
    importcpp: "convolve2(@)", header : "arrayfire.h".}

proc convolve3*(signal: Matrix; filter: Matrix; mode: ConvMode = ConvMode.DEFAULT ;
               domain: ConvDomain = ConvDomain.AUTO ): Matrix {.cdecl,
    importcpp: "convolve3(@)", header : "arrayfire.h".}

proc fftConvolve*(signal: Matrix; filter: Matrix; mode: ConvMode ): Matrix {.
    cdecl, importcpp: "fftConvolve(@)", header : "arrayfire.h".}

proc fftConvolve1*(signal: Matrix; filter: Matrix; mode: ConvMode = ConvMode.DEFAULT ): Matrix {.
    cdecl, importcpp: "fftConvolve1(@)", header : "arrayfire.h".}

proc fftConvolve2*(signal: Matrix; filter: Matrix; mode: ConvMode = ConvMode.DEFAULT ): Matrix {.
    cdecl, importcpp: "fftConvolve2(@)", header : "arrayfire.h".}

proc fftConvolve3*(signal: Matrix; filter: Matrix; mode: ConvMode = ConvMode.DEFAULT ): Matrix {.
    cdecl, importcpp: "fftConvolve3(@)", header : "arrayfire.h".}

proc fir*(b: Matrix; x: Matrix): Matrix {.cdecl, importcpp: "fir(@)", header : "arrayfire.h".}

proc iir*(b: Matrix; a: Matrix; x: Matrix): Matrix {.cdecl, importcpp: "iir(@)",
    header : "arrayfire.h".}

proc medfilt*(matin : Matrix; windLength: DimT = 3; windWidth: DimT = 3;
             edgePad: BorderType ): Matrix {.cdecl, importcpp: "medfilt(@)",
    header : "arrayfire.h".}

proc medfilt1*(matin : Matrix; windWidth: DimT = 3; edgePad: BorderType ): Matrix {.
    cdecl, importcpp: "medfilt1(@)", header : "arrayfire.h".}

proc medfilt2*(matin : Matrix; windLength: DimT = 3; windWidth: DimT = 3;
              edgePad: BorderType ): Matrix {.cdecl,
    importcpp: "medfilt2(@)", header : "arrayfire.h".}

proc sparse*(nRows: DimT; nCols: DimT; values: Matrix; rowIdx: Matrix; colIdx: Matrix;
            stype: Storage ): Matrix {.cdecl, importcpp: "sparse(@)",
    header : "arrayfire.h".}

proc sparse*(nRows: DimT; nCols: DimT; nNZ: DimT; values: pointer; rowIdx: ptr cint;
            colIdx: ptr cint; `type`: Dtype = f32; stype: Storage ;
            src: Source = Source.afHost): Matrix {.cdecl, importcpp: "sparse(@)", header : "arrayfire.h".}

proc sparse*(dense: Matrix; stype: Storage ): Matrix {.cdecl,
    importcpp: "sparse(@)", header : "arrayfire.h".}

proc sparseConvertTo*(matin : Matrix; destStrorage: Storage): Matrix {.cdecl,
    importcpp: "sparseConvertTo(@)", header : "arrayfire.h".}

proc dense*(sparse: Matrix): Matrix {.cdecl, importcpp: "dense(@)", header : "arrayfire.h".}

proc sparseGetInfo*(values: var Matrix; rowIdx: var Matrix; colIdx: var Matrix;
                   stype: var Storage; matin : Matrix) {.cdecl,
    importcpp: "sparseGetInfo(@)", header : "arrayfire.h".}

proc sparseGetValues*(matin : Matrix): Matrix {.cdecl, importcpp: "sparseGetValues(@)",
    header : "arrayfire.h".}

proc sparseGetRowIdx*(matin : Matrix): Matrix {.cdecl, importcpp: "sparseGetRowIdx(@)",
    header : "arrayfire.h".}

proc sparseGetColIdx*(matin : Matrix): Matrix {.cdecl, importcpp: "sparseGetColIdx(@)",
    header : "arrayfire.h".}

proc sparseGetNNZ*(matin : Matrix): DimT {.cdecl, importcpp: "sparseGetNNZ(@)",
                                     header : "arrayfire.h".}

proc sparseGetStorage*(matin : Matrix): Storage {.cdecl,
    importcpp: "sparseGetStorage(@)", header : "arrayfire.h".}    

proc mean*(matin : Matrix; dim: DimT ): Matrix {.cdecl, importcpp: "mean(@)",
    header : "arrayfire.h".}

proc mean*(matin : Matrix; weights: Matrix; dim: DimT ): Matrix {.cdecl,
    importcpp: "mean(@)", header : "arrayfire.h".}

proc `var`*(matin : Matrix; isbiased: bool = false; dim: DimT = - 1): Matrix {.cdecl,
    importcpp: "var(@)", header : "arrayfire.h".}

proc `var`*(matin : Matrix; weights: Matrix; dim: DimT = - 1): Matrix {.cdecl,
    importcpp: "var(@)", header : "arrayfire.h".}

proc topk*(values: Matrix, indices: Matrix, matin: Matrix, k: cint, dim: cint = - 1, order: TopKFunction = TopKFunction.TOPK_MAX) {.cdecl,
    importcpp: "topk(@)", header : "arrayfire.h".}

proc stdev*(matin : Matrix; dim: DimT ): Matrix {.cdecl, importcpp: "stdev(@)",
    header : "arrayfire.h".}

proc cov*(x: Matrix; y: Matrix; isbiased: bool = false): Matrix {.cdecl,
    importcpp: "cov(@)", header : "arrayfire.h".}

proc median*(matin : Matrix; dim: DimT): Matrix {.cdecl, importcpp: "median(@)",
    header : "arrayfire.h".}

proc mean*(matin : Matrix): cdouble 
  {.cdecl, importcpp: "af::mean<double>(@)", header : "arrayfire.h".}

proc mean*[T](matin : Matrix; weights: Matrix): T 
  {.cdecl, importcpp: "af::mean<'*0>(@)",header : "arrayfire.h".}

proc `var`*[T](matin : Matrix; isbiased: bool = false): T 
  {.cdecl, importcpp: "af::var<'*0>(@)",header : "arrayfire.h".}

proc `var`*(matin : Matrix; weights: Matrix): cdouble
  {.cdecl, importcpp: "var<double>(@)",header : "arrayfire.h".}

proc stdev*(matin : Matrix): cdouble
  {.cdecl, importcpp: "af::stdev<double>(@)", header : "arrayfire.h".}

proc median*(matin : Matrix): cdouble
  {.cdecl, importcpp: "af::median<double>(@)", header : "arrayfire.h".}

proc corrcoef*(x: Matrix; y: Matrix): cdouble
  {.cdecl, importcpp: "af::corrcoef<double>(@)",header : "arrayfire.h".}

proc print*(exp: cstring; arr: Matrix) {.cdecl, importcpp: "print(@)", header : "arrayfire.h".}

proc print*(exp: cstring; arr: Matrix; precision: cint) {.cdecl, importcpp: "print(@)",
    header : "arrayfire.h".}

proc saveArray*(key: cstring; arr: Matrix; filename: cstring; append: bool = false): cint 
  {.cdecl, importcpp: "af::saveArray(@)", header : "arrayfire.h".}

proc readArray*(filename: cstring; index: cuint): Matrix 
  {.cdecl,importcpp: "af::readArray(@)", header : "arrayfire.h".}

proc readArray*(filename: cstring; key: cstring): Matrix 
  {.cdecl, importcpp: "af::readArray(@)", header : "arrayfire.h".}

proc readArrayCheck*(filename: cstring; key: cstring): cint {.cdecl,
    importcpp: "af::readArrayCheck(@)", header : "arrayfire.h".}

proc toString*(output: cstringArray; exp: cstring; arr: Matrix; precision: cint = 4;
              transpose: bool = true) 
  {.cdecl, importcpp: "af::toString(@)", header : "arrayfire.h".}

proc toString*(exp: cstring; arr: Matrix; precision: cint = 4; transpose: bool = true): cstring {.
    cdecl, importcpp: "af::toString(@)", header : "arrayfire.h".}

proc exampleFunction*(matin : Matrix; param: SomeenumT): Matrix {.cdecl,
    importcpp: "exampleFunction(@)", header : "arrayfire.h".}

proc getSizeOf*(`type`: Dtype): csize_t {.cdecl, importcpp: "getSizeOf(@)", header : "arrayfire.h".}

proc fast*(matin : Matrix; thr: cfloat = 20.0; arcLength: cuint = 9; nonMax: bool = true;
          featureRatio: cfloat = 0.05; edge: cuint = 3): Features {.cdecl,
    importcpp: "fast(@)", header : "arrayfire.h".}

proc harris*(matin : Matrix; maxCorners: cuint = 500; minResponse: cfloat = 100000.0;
            sigma: cfloat = 1.0; blockSize: cuint = 0; kThr: cfloat = 0.04): Features {.
    cdecl, importcpp: "harris(@)", header : "arrayfire.h".}

proc orb*(feat: var Features; desc: var Matrix; image: Matrix; fastThr: cfloat = 20.0;
         maxFeat: cuint = 400; sclFctr: cfloat = 1.5; levels: cuint = 4;
         blurImg: bool = false) {.cdecl, importcpp: "orb(@)", header : "arrayfire.h".}

proc sift*(feat: var Features; desc: var Matrix; matin : Matrix; nLayers: cuint = 3;
          contrastThr: cfloat = 0.04; edgeThr: cfloat = 10.0; initSigma: cfloat = 1.6;
          doubleInput: bool = true; intensityScale: cfloat = 0.00390625;
          featureRatio: cfloat = 0.05) {.cdecl, importcpp: "sift(@)", header : "arrayfire.h".}

proc gloh*(feat: var Features; desc: var Matrix; matin : Matrix; nLayers: cuint = 3;
          contrastThr: cfloat = 0.04; edgeThr: cfloat = 10.0; initSigma: cfloat = 1.6;
          doubleInput: bool = true; intensityScale: cfloat = 0.00390625;
          featureRatio: cfloat = 0.05) {.cdecl, importcpp: "gloh(@)", header : "arrayfire.h".}

proc hammingMatcher*(idx: var Matrix; dist: var Matrix; query: Matrix; train: Matrix;
                    distDim: DimT = 0; nDist: cuint = 1) {.cdecl,
    importcpp: "hammingMatcher(@)", header : "arrayfire.h".}

proc nearestNeighbour*(idx: var Matrix; dist: var Matrix; query: Matrix; train: Matrix;
                      distDim: DimT = 0; nDist: cuint = 1; distType: MatchType ) {.
    cdecl, importcpp: "nearestNeighbour(@)", header : "arrayfire.h".}

proc matchTemplate*(searchImg: Matrix; templateImg: Matrix; mType: MatchType ): Matrix {.
    cdecl, importcpp: "matchTemplate(@)", header : "arrayfire.h".}

proc susan*(matin : Matrix; radius: cuint = 3; diffThr: cfloat = 32.0;
           geomThr: cfloat = 10.0; featureRatio: cfloat = 0.05; edge: cuint = 3): Features {.
    cdecl, importcpp: "susan(@)", header : "arrayfire.h".}

proc dog*(matin : Matrix; radius1: cint; radius2: cint): Matrix {.cdecl,
    importcpp: "dog(@)", header : "arrayfire.h".}

proc homography*(h: var Matrix; inliers: var cint; xSrc: Matrix; ySrc: Matrix;
                xDst: Matrix; yDst: Matrix;
                htype: HomographyType ; inlierThr: cfloat = 3.0;
                iterations: cuint = 1000; otype: Dtype = f32) {.cdecl,
    importcpp: "homography(@)", header : "arrayfire.h".}  

proc constructindex*(): Index {.constructor, importcpp: "index(@)",
                             header: "arrayfire.h".}
proc destroyindex*(this: var Index) {.importcpp: "#.~index()", header: "arrayfire.h".}
proc constructindex*(idx: cint): Index {.constructor, importcpp: "index(@)",
                                     header: "arrayfire.h".}

proc constructindex*(s0: AF_Seq): Index {.constructor, importcpp: "af::index(@)",
                                       header: "arrayfire.h".}
proc constructindex*(idx0: Matrix): Index {.constructor, importcpp: "af::index(@)",
                                        header: "arrayfire.h".}
proc constructindex*(idx0: Index): Index {.constructor, importcpp: "af::index(@)",
                                       header: "arrayfire.h".}
proc isspan*(this: Index): bool {.noSideEffect, importcpp: "isspan",
                              header: "arrayfire.h".}
proc get*(this: Index): IndexT {.noSideEffect, importcpp: "get", header: "arrayfire.h".}
proc constructindex*(idx0: var AF_Seq): Index {.constructor, importcpp: "index(@)",
    header: "arrayfire.h".}

proc constructindex*(idx0: var Matrix): Index {.constructor, importcpp: "index(@)",
    header: "arrayfire.h".}


proc lookup*(matin: Matrix; idx: Matrix; dim: cint = - 1): Matrix {.importcpp: "lookup(@)",
    header: "arrayfire.h".}

let span* = constructindex(mseq(1,1,0))


proc copy*(dst: var Matrix; src: Matrix; idx0: Index; idx1: Index = span;
          idx2: Index = span; idx3: Index = span) {.importcpp: "copy(@)",
    header: "arrayfire.h".}


proc constructarrayProxy*(par: var Matrix; ssss: ptr IndexT; linear: bool = false): Matrix_View {.
    constructor, importcpp: "array_proxy(@)", header: "arrayfire.h".}
proc constructarrayProxy*(other: Matrix_View): Matrix_View {.constructor,
    importcpp: "array_proxy(@)", header: "arrayfire.h".}
proc constructarrayProxy*(other: var Matrix_View): Matrix_View {.constructor,
    importcpp: "array_proxy(@)", header: "arrayfire.h".}
proc destroyarrayProxy*(this: var Matrix_View) {.importcpp: "#.~array_proxy()",
    header: "arrayfire.h".}

proc toMatrix*(this: Matrix_View) 
  {.noSideEffect, importcpp: "array", header: "arrayfire.h".}

proc toMatrix*(this: var Matrix_View) 
  {.importcpp: "array", header: "arrayfire.h".}


proc assign*(this: var Matrix_View; a: Matrix) {.importcpp: "#.operator=(@)", header: "arrayfire.h".}
proc assign*(this: var Matrix_View; a: cdouble) {.importcpp: "#.operator=(@)",header: "arrayfire.h".}
proc assign*(this: var Matrix_View; a: cfloat) {.importcpp: "#.operator=(@)",header: "arrayfire.h".}
proc assign*(this: var Matrix_View; a: cint) {.importcpp: "#.operator=(@)",header: "arrayfire.h".}
proc assign*(this: var Matrix_View; a: cuint) {.importcpp: "#.operator=(@)",header: "arrayfire.h".}
proc assign*(this: var Matrix_View; a: bool) {.importcpp: "#.operator=(@)",header: "arrayfire.h".}
proc assign*(this: var Matrix_View; a: char) {.importcpp: "#.operator=(@)",header: "arrayfire.h".}
proc assign*(this: var Matrix_View; a: clong) {.importcpp: "#.operator=(@)",header: "arrayfire.h".}
proc assign*(this: var Matrix_View; a: culong) {.importcpp: "#.operator=(@)",header: "arrayfire.h".}
proc assign*(this: var Matrix_View; a: clonglong) {.importcpp: "#.operator=(@)",header: "arrayfire.h".}
proc assign*(this: var Matrix_View; a: culonglong) {.importcpp: "#.operator=(@)",header: "arrayfire.h".}


proc `+=`*(this: var Matrix_View; a: Matrix) 
  {.importcpp: "(# += #)", header: "arrayfire.h".}

proc `+=`*(this: var Matrix_View; a: cdouble) 
  {.importcpp: "(# += #)",header: "arrayfire.h".}

proc `+=`*(this: var Matrix_View; a: cfloat) 
  {.importcpp: "(# += #)",header: "arrayfire.h".}

proc `+=`*(this: var Matrix_View; a: cint) 
  {.importcpp: "(# += #)",header: "arrayfire.h".}

proc `+=`*(this: var Matrix_View; a: cuint) 
  {.importcpp: "(# += #)",header: "arrayfire.h".}

proc `+=`*(this: var Matrix_View; a: bool) 
  {.importcpp: "(# += #)",header: "arrayfire.h".}

proc `+=`*(this: var Matrix_View; a: char) 
  {.importcpp: "(# += #)",header: "arrayfire.h".}

proc `+=`*(this: var Matrix_View; a: clong) 
  {.importcpp: "(# += #)",header: "arrayfire.h".}

proc `+=`*(this: var Matrix_View; a: culong) 
  {.importcpp: "(# += #)",header: "arrayfire.h".}

proc `+=`*(this: var Matrix_View; a: clonglong) {.importcpp: "(# += #)",
    header: "arrayfire.h".}
proc `+=`*(this: var Matrix_View; a: culonglong) {.importcpp: "(# += #)",
    header: "arrayfire.h".}
proc `-=`*(this: var Matrix_View; a: Matrix_View) {.importcpp: "(# -= #)",
    header: "arrayfire.h".}
proc `-=`*(this: var Matrix_View; a: cdouble) {.importcpp: "(# -= #)",
    header: "arrayfire.h".}
proc `-=`*(this: var Matrix_View; a: cfloat) {.importcpp: "(# -= #)",
    header: "arrayfire.h".}
proc `-=`*(this: var Matrix_View; a: cint) {.importcpp: "(# -= #)",
                                       header: "arrayfire.h".}
proc `-=`*(this: var Matrix_View; a: cuint) {.importcpp: "(# -= #)",
                                        header: "arrayfire.h".}
proc `-=`*(this: var Matrix_View; a: bool) {.importcpp: "(# -= #)",
                                       header: "arrayfire.h".}
proc `-=`*(this: var Matrix_View; a: char) {.importcpp: "(# -= #)",
                                       header: "arrayfire.h".}
proc `-=`*(this: var Matrix_View; a: clong) {.importcpp: "(# -= #)",
                                        header: "arrayfire.h".}
proc `-=`*(this: var Matrix_View; a: culong) {.importcpp: "(# -= #)",
    header: "arrayfire.h".}
proc `-=`*(this: var Matrix_View; a: clonglong) {.importcpp: "(# -= #)",
    header: "arrayfire.h".}
proc `-=`*(this: var Matrix_View; a: culonglong) {.importcpp: "(# -= #)",
    header: "arrayfire.h".}
proc `*=`*(this: var Matrix_View; a: Matrix_View) {.importcpp: "(# *= #)",
    header: "arrayfire.h".}
proc `*=`*(this: var Matrix_View; a: cdouble) {.importcpp: "(# *= #)",
    header: "arrayfire.h".}
proc `*=`*(this: var Matrix_View; a: cfloat) {.importcpp: "(# *= #)",
    header: "arrayfire.h".}
proc `*=`*(this: var Matrix_View; a: cint) {.importcpp: "(# *= #)",
                                       header: "arrayfire.h".}
proc `*=`*(this: var Matrix_View; a: cuint) {.importcpp: "(# *= #)",
                                        header: "arrayfire.h".}
proc `*=`*(this: var Matrix_View; a: bool) {.importcpp: "(# *= #)",
                                       header: "arrayfire.h".}
proc `*=`*(this: var Matrix_View; a: char) {.importcpp: "(# *= #)",
                                       header: "arrayfire.h".}
proc `*=`*(this: var Matrix_View; a: clong) {.importcpp: "(# *= #)",
                                        header: "arrayfire.h".}
proc `*=`*(this: var Matrix_View; a: culong) {.importcpp: "(# *= #)",
    header: "arrayfire.h".}
proc `*=`*(this: var Matrix_View; a: clonglong) {.importcpp: "(# *= #)",
    header: "arrayfire.h".}
proc `*=`*(this: var Matrix_View; a: culonglong) {.importcpp: "(# *= #)",
    header: "arrayfire.h".}
proc `/=`*(this: var Matrix_View; a: Matrix_View) {.importcpp: "(# /= #)",
    header: "arrayfire.h".}
proc `/=`*(this: var Matrix_View; a: cdouble) {.importcpp: "(# /= #)",
    header: "arrayfire.h".}
proc `/=`*(this: var Matrix_View; a: cfloat) {.importcpp: "(# /= #)",
    header: "arrayfire.h".}
proc `/=`*(this: var Matrix_View; a: cint) {.importcpp: "(# /= #)",
                                       header: "arrayfire.h".}
proc `/=`*(this: var Matrix_View; a: cuint) {.importcpp: "(# /= #)",
                                        header: "arrayfire.h".}
proc `/=`*(this: var Matrix_View; a: bool) {.importcpp: "(# /= #)",
                                       header: "arrayfire.h".}
proc `/=`*(this: var Matrix_View; a: char) {.importcpp: "(# /= #)",
                                       header: "arrayfire.h".}
proc `/=`*(this: var Matrix_View; a: clong) {.importcpp: "(# /= #)",
                                        header: "arrayfire.h".}
proc `/=`*(this: var Matrix_View; a: culong) {.importcpp: "(# /= #)",
    header: "arrayfire.h".}
proc `/=`*(this: var Matrix_View; a: clonglong) {.importcpp: "(# /= #)",
    header: "arrayfire.h".}
proc `/=`*(this: var Matrix_View; a: culonglong) {.importcpp: "(# /= #)",
    header: "arrayfire.h".}
proc get*(this: var Matrix_View): array {.importcpp: "get", header: "arrayfire.h".}

proc get*(this: Matrix_View): array {.noSideEffect, importcpp: "get",
                                  header: "arrayfire.h".}
proc dims*[M: Matrix | Matrix_View](this: M): Dim4 
  {.noSideEffect, importcpp: "dims",header: "arrayfire.h".}

proc dims*[M: Matrix | Matrix_View](this: M; dim: cuint): DimT 
  {.noSideEffect, importcpp: "dims",header: "arrayfire.h".}

proc msum*(matin: Matrix; dim: cint = -1 ): Matrix 
  {.importcpp: "af::sum(@)",header: "arrayfire.h".}

proc msum*(matin: Matrix; dim: cint; nanval: cdouble): Matrix 
  {.importcpp: "af::sum(@)",header: "arrayfire.h".}

proc sumByKey*(keys_out: Matrix, vals_out: Matrix, keys: Matrix, vals: Matrix, dim: cint = -1 ) 
  {.importcpp: "sumByKey(@)",header: "arrayfire.h".}

proc product*(matin: Matrix; dim: cint = - 1): Matrix 
  {.importcpp: "af::product(@)",header: "arrayfire.h".}

proc productByKey*(keys_out: Matrix, vals_out : Matrix, keys: Matrix, vals: Matrix, dims: cint = -1) 
  {.importcpp: "productByKey(@)",header: "arrayfire.h".}  

proc product*(matin: Matrix; dim: cint; nanval: cdouble): Matrix 
  {.importcpp: "af::product(@)", header: "arrayfire.h".}
proc mmin*(matin: Matrix; dim: cint = - 1): Matrix {.importcpp: "af::min(@)",
    header: "arrayfire.h".}

proc maxByKey*(keys_out: Matrix; vals_out: Matrix, keys: Matrix, vals: Matrix; dim: cint = - 1) {.importcpp: "maxByKey(@)",
    header: "arrayfire.h".}

proc minByKey*(keys_out: Matrix; vals_out: Matrix, keys: Matrix, vals: Matrix; dim: cint = - 1) {.importcpp: "minByKey(@)",
    header: "arrayfire.h".}

proc mmax*(matin: Matrix; dim: cint = - 1): Matrix {.importcpp: "af::max(@)",
    header: "arrayfire.h".}

proc allTrue*(matin: Matrix; dim: cint = - 1): Matrix {.importcpp: "af::allTrue(@)",
    header: "arrayfire.h".}

proc allTrueByKey*(keys_out: Matrix; vals_out: Matrix, keys: Matrix, vals: Matrix; dim: cint = - 1) {.importcpp: "allTrueByKey(@)",
    header: "arrayfire.h".}

proc anyTrue*(matin: Matrix; dim: cint = - 1): Matrix {.importcpp: "af::anyTrue(@)",
    header: "arrayfire.h".}

proc anyTrueByKey*(keys_out: Matrix; vals_out: Matrix, keys: Matrix, vals: Matrix; dim: cint = - 1) {.importcpp: "anyTrueByKey(@)",
    header: "arrayfire.h".}

proc count*(matin: Matrix; dim: cint = - 1): Matrix {.importcpp: "af::count(@)",
    header: "arrayfire.h".}

proc countByKey*(keys_out: var Matrix; vals_out: var Matrix; keys: Matrix, vals: Matrix) {.importcpp: "countByKey(@)",
    header: "arrayfire.h".}

proc af_sum_all(real : ptr[cdouble], imag : ptr[cdouble], carray : AF_Array_Handle) : Err 
    {.importcpp: "af_sum_all(@)",header: "arrayfire.h".}

proc s_native_sum*(matin: Matrix) : tuple[real: float, imag: float] =
  var real : cdouble = 0
  var imag : cdouble = 0
  discard af_sum_all(addr real, addr imag, matin.get())
  (real,imag)

proc sum_as_int*(matin: Matrix) : int =
  int(s_native_sum(matin)[0])

proc sum_as_float*(matin: Matrix) : float =
  float(s_native_sum(matin)[0])

proc sum_as_complex*(matin: Matrix) : Complex64 =
  var (real,imag) = s_native_sum(matin)
  complex64(real,imag)

proc af_product_all(real : ptr[cdouble], imag : ptr[cdouble], carray : AF_Array_Handle) : Err 
    {.importcpp: "af_product_all(@)",header: "arrayfire.h".}

proc s_native_product*(matin: Matrix) : tuple[real: float, imag: float] =
  var real : cdouble = 0
  var imag : cdouble = 0
  discard af_product_all(addr real, addr imag, matin.get())
  (real,imag)

proc product_as_int*(matin: Matrix) : int =
  int(s_native_product(matin)[0])

proc product_as_float*(matin: Matrix) : float =
  float(s_native_product(matin)[0])

proc product_as_complex*(matin: Matrix) : Complex64 =
  var (real,imag) = s_native_product(matin)
  complex64(real,imag)


proc af_min_all(real : ptr[cdouble], imag : ptr[cdouble], carray : AF_Array_Handle) : Err 
    {.importcpp: "af_min_all(@)",header: "arrayfire.h".}

proc s_native_min*(matin: Matrix) : tuple[real: float, imag: float] =
  var real : cdouble = 0
  var imag : cdouble = 0
  discard af_min_all(addr real, addr imag, matin.get())
  (real,imag)

proc min_as_int*(matin: Matrix) : int =
  int(s_native_min(matin)[0])

proc min_as_float*(matin: Matrix) : float =
  float(s_native_min(matin)[0])

proc min_as_complex*(matin: Matrix) : Complex64 =
  var (real,imag) = s_native_min(matin)
  complex64(real,imag)


proc af_max_all(real : ptr[cdouble], imag : ptr[cdouble], carray : AF_Array_Handle) : Err 
    {.importcpp: "af_max_all(@)",header: "arrayfire.h".}

proc s_native_max*(matin: Matrix) : tuple[real: float, imag: float] =
  var real : cdouble = 0
  var imag : cdouble = 0
  discard af_max_all(addr real, addr imag, matin.get())
  (real,imag)

proc max_as_int*(matin: Matrix) : int =
  int(s_native_max(matin)[0])

proc max_as_float*(matin: Matrix) : float =
  float(s_native_max(matin)[0])

proc max_as_complex*(matin: Matrix) : Complex64 =
  var (real,imag) = s_native_max(matin)
  complex64(real,imag)


proc s_allTrue*(matin: Matrix): bool
  {.importcpp: "af::allTrue<bool>(@)", header: "arrayfire.h".}

proc s_anyTrue*(matin: Matrix): bool
  {.importcpp: "af::anyTrue<bool>(@)", header: "arrayfire.h".}

proc count*(matin: Matrix): cuint
  {.importcpp: "af::count<uint>(@)", header: "arrayfire.h".}

proc min*(val: var Matrix; idx: var Matrix; matin: Matrix; dim: cint = - 1) 
  {.importcpp: "af::min(@)", header: "arrayfire.h".}

proc max*(val: var Matrix; idx: var Matrix; matin: Matrix; dim: cint = - 1) 
  {.importcpp: "af::max(@)", header: "arrayfire.h".}

proc min*[T](val: ptr T; idx: ptr cuint; matin: Matrix) 
  {.importcpp: "af::min(@)",header: "arrayfire.h".}

proc max*[T](val: ptr T; idx: ptr cuint; matin: Matrix) 
  {.importcpp: "af::max(@)",header: "arrayfire.h".}

proc accum*(matin: Matrix; dim: cint = 0): Matrix 
  {.importcpp: "af::accum(@)",header: "arrayfire.h".}

proc scan*(matin: Matrix; dim: cint = 0, op: BinaryOp = BinaryOp.BINARY_ADD, inclusiveScan: bool = true): Matrix 
  {.importcpp: "af::scan(@)",header: "arrayfire.h".}

proc scanByKey*(key: Matrix, matin: Matrix, dim: cint = 0, op: BinaryOp = BinaryOp.BINARY_ADD, inclusiveScan: bool = true): Matrix 
  {.importcpp: "af::scanByKey(@)",header: "arrayfire.h".}

proc where*(matin: Matrix): Matrix 
  {.importcpp: "af::where(@)", header: "arrayfire.h".}

proc diff1*(matin: Matrix; dim: cint = 0): Matrix 
  {.importcpp: "af::diff1(@)",header: "arrayfire.h".}

proc diff2*(matin: Matrix; dim: cint = 0): Matrix 
  {.importcpp: "af::diff2(@)",header: "arrayfire.h".}

proc sort*(matin: Matrix; dim: cuint = 0; isAscending: bool = true): Matrix 
  {.importcpp: "af::sort(@)", header: "arrayfire.h".}

proc sort*(`out`: var Matrix; indices: var Matrix; matin: Matrix; dim: cuint = 0;
          isAscending: bool = true) 
  {.importcpp: "af::sort(@)", header: "arrayfire.h".}

proc sortByKeys*(outKeys: var Matrix; outValues: var Matrix; keys: Matrix; values: Matrix;
          dim: cuint = 0; isAscending: bool = true) 
  {.importcpp: "sortByKeys(@)",header: "arrayfire.h".}

proc setUnique*(matin: Matrix; isSorted: bool = false): Matrix 
  {.importcpp: "af::setUnique(@)", header: "arrayfire.h".}

proc setUnion*(first: Matrix; second: Matrix; isUnique: bool = false): Matrix 
  {.importcpp: "af::setUnion(@)", header: "arrayfire.h".}

proc setIntersect*(first: Matrix; second: Matrix; isUnique: bool = false): Matrix 
  {.importcpp: "af::setIntersect(@)", header: "arrayfire.h".}    

proc af_timeit*(fn: proc ()): cdouble {.importcpp: "af::timeit(@)", header: "arrayfire.h".}

proc timer_start*(): Timer 
  {.importcpp: "af::timer::start(@)", header: "arrayfire.h".}

proc timer_stop*(): cdouble 
  {.importcpp: "af::timer::stop(@)", header: "arrayfire.h".}

proc timer_stop*(start: Timer): cdouble 
  {.importcpp: "af::timer::stop(@)", header: "arrayfire.h".}

# ----------------- Syntactic Sugar -----------------

const dtype_map = {
  "int":sysint,
  "uint":sysuint,
  "float":sysfloat,
  "Complex32" : Dtype.c32,
  "Complex64" : Dtype.c64,
  "float32":Dtype.f32,
  "float64":Dtype.f64,
  "int16":Dtype.s16,
  "int32":Dtype.s32,
  "int64":Dtype.s64,
  "uint8":Dtype.u8,
  "uint16":Dtype.u16,
  "uint32":Dtype.u32,
  "uint64":Dtype.u64,
  "bool":Dtype.b8
  }.toTable

proc get_DType*(tname: string) : DType =
  result=dtype_map[tname]

const dtype_size_map = {
  DType.f32 : sizeof(float32),
  DType.c32 : sizeof(float32)*2,
  DType.f64 : sizeof(float64),
  DType.c64 : sizeof(float64)*2,
  DType.b8  : sizeof(bool),
  DType.s32 : sizeof(int32),
  DType.u32 : sizeof(uint32),
  DType.u8  : sizeof(uint8),
  DType.s64 : sizeof(int64),
  DType.u64 : sizeof(uint64),
  DType.s16 : sizeof(int16),
  DType.u16 : sizeof(uint16),
}.toTable



proc get_DType_size*(at:DType) : int =
  result=dtype_size_map[at]            

let iend* : cint = -1

converter toMatrix*(s: AF_SEQ) : Matrix = s.get_Matrix()

converter toMatrix*(mv: Matrix_View) : Matrix = mv.get_Matrix()

converter toCuint*(d: DimT) : cuint = cuint(d)

converter toInt*(i: cint) : int = int(i)

converter toInt*(i: clong) : int = int(i)

proc copy_array_to_c[T](data:openarray[T]) : pointer =  
  doAssert len(data) > 0 
  result = alloc0(data.len*sizeof(T))   
  for i in 0..<data.len:
    var target_ptr=cast[ptr T](cast[int](result) + (i * sizeof(T)))
    target_ptr[]=data[i]

proc matrix*[T](dims : Dim4, data : openarray[T]) : Matrix =
  when (T is int): 
    var cc = newSeq[int32]()
    for i in data: 
      cc.add(int32(i))
    let cdata = copy_array_to_c(cc)
    result = matrix[int32](dims,cast[ptr int32](cdata))
  else:  
    let cdata = copy_array_to_c(data)
    result = matrix[T](dims,cast[ptr T](cdata))
  dealloc(cdata)

proc matrix*[T](dims : Dim4, data : openarray[T], matrix_type : DType) : Matrix =
  let tmp = matrix(dims,data)
  tmp.`as`(matrix_type)

proc matrix*[T](dim0 : DimT, data : openarray[T]) : Matrix =
  matrix(dim4(dim0),data)

proc matrix*[T](dim0 : DimT, dim1 : DimT, data : openarray[T]) : Matrix =
  matrix(dim4(dim0,dim1),data)

proc matrix*[T](dim0 : DimT, dim1 : DimT, dim2 : DimT, data : openarray[T]) : Matrix =
  matrix(dim4(dim0,dim1,dim2),data)

proc matrix*[T](dim0 : DimT, dim1 : DimT, dim2 : DimT, dim3 : DimT, data : openarray[T]) : Matrix =
  matrix(dim4(dim0,dim1,dim2,dim3),data)

proc matrix*[T](dim0 : DimT, data : openarray[T], matrix_type : DType) : Matrix =
  matrix(dim4(dim0),data,matrix_type)

proc matrix*[T](dim0 : DimT, dim1 : DimT, data : openarray[T], matrix_type : DType) : Matrix =
  matrix(dim4(dim0,dim1),data,matrix_type)

proc matrix*[T](dim0 : DimT, dim1 : DimT, dim2 : DimT, data : openarray[T], 
    matrix_type : DType) : Matrix =
  matrix(dim4(dim0,dim1,dim2),data, matrix_type)

proc matrix*[T](dim0 : DimT, dim1 : DimT, dim2 : DimT, dim3 : DimT, data : openarray[T], 
    matrix_type : DType) : Matrix =
  matrix(dim4(dim0,dim1,dim2,dim3),data, matrix_type)


proc matrix*[T](dims : Dim4, slice : Slice[T]) : Matrix =
  var data: seq[int] = @[]
  for i in slice.a..slice.b:
    data.add(i)
  when(T is int): 
    var cc = newSeq[int32]()
    for i in data: 
      cc.add(int32(i))
    let cdata = copy_array_to_c(cc)
    result = matrix[int32](dims,cast[ptr int32](cdata), src=Source.afHost)
  else:
    let cdata = copy_array_to_c(data)
    result = matrix[T](dims,cast[ptr T](cdata), src=Source.afHost)
  dealloc(cdata)


proc matrix*[T](dims : Dim4, slice : Slice[T], matrix_type : DType) : Matrix =
  let tmp = matrix(dims,slice)
  tmp.`as`(matrix_type)

proc matrix*[T](dim0 : DimT, data : Slice[T]) : Matrix =
  matrix(dim4(dim0),data)

proc matrix*[T](dim0 : DimT, dim1 : DimT, data : Slice[T]) : Matrix =
  matrix(dim4(dim0,dim1),data)

proc matrix*[T](dim0 : DimT, dim1 : DimT, dim2 : DimT, data : Slice[T]) : Matrix =
  matrix(dim4(dim0,dim1,dim2),data)

proc matrix*[T](dim0 : DimT, dim1 : DimT, dim2 : DimT, dim3 : DimT, data : Slice[T]) : Matrix =
  matrix(dim4(dim0,dim1,dim2,dim3),data)

proc matrix*[T](dim0 : DimT, data : Slice[T], matrix_type : DType) : Matrix =
  matrix(dim4(dim0),data,matrix_type)

proc matrix*[T](dim0 : DimT, dim1 : DimT, data : Slice[T], matrix_type : DType) : Matrix =
  matrix(dim4(dim0,dim1),data,matrix_type)

proc matrix*[T](dim0 : DimT, dim1 : DimT, dim2 : DimT, data : Slice[T], 
    matrix_type : DType) : Matrix =
  matrix(dim4(dim0,dim1,dim2),data, matrix_type)

proc matrix*[T](dim0 : DimT, dim1 : DimT, dim2 : DimT, dim3 : DimT, data : Slice[T], 
    matrix_type : DType) : Matrix =
  matrix(dim4(dim0,dim1,dim2,dim3),data, matrix_type)

proc ndims*(m: Matrix) : DimT =
  m.dims.ndims

proc `$`*(m: Matrix) : string =
  result = $toString("",m)

proc randu*(dims: openarray[int],ty: Dtype = f32) : Matrix =
  randu(dim4s(dims),ty)

proc randn*(dims: openarray[int],ty: Dtype = f32) : Matrix =
  randn(dim4s(dims),ty)

proc mseq*[T1:int | float | int32 | int64](last: T1) : AF_Seq = 
  mseq(cdouble(last))

proc mseq*[T1:int | float | int32 | int64,
           T2:int | float | int32 | int64](first: T1, last: T2) : AF_Seq = 
  mseq(cdouble(first), cdouble(last))

proc mseq*[T1:int | float | int32 | int64,
           T2:int | float | int32 | int64,
           T3:int | float | int32 | int64,](first: T1, last: T2, step: T3) = 
  mseq(cdouble(first), cdouble(last),cdouble(step))

proc begin(this: AF_Seq) : cdouble
  {.importcpp: "#.s.begin", header: "arrayfire.h".}

proc until(this: AF_Seq) : cdouble
  {.importcpp: "#.s.end", header: "arrayfire.h".}

proc step(this: AF_Seq) : cdouble
  {.importcpp: "#.s.step", header: "arrayfire.h".}

proc getContent*(m: AF_Seq) : tuple[start: float, until: float, step: float] =
  (m.begin,m.until,m.step)

proc `$`*(m: AF_Seq) : string =
  let vals = m.getContent
  "AF_Seq[from: $1, until $2, step: $3]"%[$vals[0],$vals[1],$vals[2]]

proc `[]`*[I: int | int64 | AF_Seq | Matrix | Matrix_View | Index,  
           M: Matrix | Matrix_View](m: M, i: I) : Matrix_View 
  {.importcpp: "#(#)", header: "arrayfire.h".}

proc `[]`*[I1: int | int64 | AF_Seq | Matrix | Matrix_View | Index, 
          I2: int | int64 | AF_Seq | Matrix | Matrix_View | Index,
          M: Matrix | Matrix_View](m: M, i1: I1, i2 : I2) : Matrix_View 
  {.importcpp: "#(@)", header: "arrayfire.h".}

proc `[]`*[I1: int | int64 | AF_Seq | Matrix | Matrix_View | Index,
           I2: int | int64 | AF_Seq | Matrix | Matrix_View | Index,
           I3: int | int64 | AF_Seq | Matrix | Matrix_View | Index,
           M: Matrix | Matrix_View
           ](m: M, idx1: I1, idx2 : I2, idx3 : I3) : Matrix_View =
  {.importcpp: "#(@)", header: "arrayfire.h".}


proc `[]`*[I1: int | int64 | AF_Seq | Matrix | Matrix_View | Index,
           I2: int | int64 | AF_Seq | Matrix | Matrix_View | Index,
           I3: int | int64 | AF_Seq | Matrix | Matrix_View | Index,
           I4: int | int64 | AF_Seq | Matrix | Matrix_View | Index,
           M: Matrix | Matrix_View
           ](m: M, idx1: I1, idx2 : I2, idx3 : I3, idx4 : I4 ) : Matrix_View =
  {.importcpp: "#(@)", header: "arrayfire.h".}


proc `[]=`*[I1: int | int64 | AF_Seq | Index | Matrix | Matrix_View, 
            V:cdouble | cfloat | cint | cuint | clong | culong | clonglong | 
              culonglong | char | bool | Matrix | Matrix_View | Matrix_View,
            M: Matrix | Matrix_View          
  ](this: var M; idx1: I1, val: V) 
  {.importcpp: "#(#).operator=(@)", header: "arrayfire.h".}


proc `[]=`*[I1: int | int64 | AF_Seq | Index | Matrix | Matrix_View, 
            I2: int | int64 | AF_Seq | Index | Matrix | Matrix_View, 
            V:cdouble | cfloat | cint | cuint | clong | culong | clonglong | 
              culonglong | char | bool | Matrix | Matrix_View | Matrix_View,
            M: Matrix | Matrix_View
  ](this: var M; idx1: I1, idx2 : I2, val: V) 
  {.importcpp: "#(#,#).operator=(@)", header: "arrayfire.h".}


proc `[]=`*[I1: int | int64 | AF_Seq | Index | Matrix | Matrix_View, 
            I2: int | int64 | AF_Seq | Index | Matrix | Matrix_View, 
            I3: int | int64 | AF_Seq | Index | Matrix | Matrix_View, 
            V:cdouble | cfloat | cint | cuint | clong | culong | clonglong | 
              culonglong | char | bool | Matrix | Matrix_View | Matrix_View,
            M: Matrix | Matrix_View
  ](this: var M; idx1: I1, idx2 : I2, idx3 : I3, val: V) 
  {.importcpp: "#(#,#,#).operator=(@)", header: "arrayfire.h".}


proc `[]=`*[I1: int | int64 | AF_Seq | Index | Matrix | Matrix_View, 
            I2: int | int64 | AF_Seq | Index | Matrix | Matrix_View, 
            I3: int | int64 | AF_Seq | Index | Matrix | Matrix_View, 
            I4: int | int64 | AF_Seq | Index | Matrix | Matrix_View, 
            V:cdouble | cfloat | cint | cuint | clong | culong | clonglong | 
              culonglong | char | bool | Matrix | Matrix_View | Matrix_View,
            M: Matrix | Matrix_View
  ](this: var M; idx1: I1, idx2 : I2, idx3 : I3, idx4 : I4, val: V) 
  {.importcpp: "#(#,#,#,#).operator=(@)", header: "arrayfire.h".}


proc len*(m: Matrix) : int =
  int(m.elements())


template timeit*(time: var float, actions: untyped)  =
  let t0 = timer_start()
  actions
  time = timer_stop(t0)

template gfor*(s, last, actions: untyped) =
  var s = mseq(mseq(cdouble(last)),true)
  while(gforToggle()):
    actions

template gfor*(s, first, last, actions: untyped) =
  s = mseq(mseq(cdouble(first),cdouble(last)),true)
  while(gforToggle()):
    actions

template gfor*(s, first, last, step, actions: untyped) =
  s = mseq(mseq(cdouble(first),cdouble(last),cdouble(step)),true) 
  while(gforToggle()):
    actions

template window*(wvar : untyped, width : int, height : int, title : string) =
  var wvar : Window
  wvar.setSize(width,height)
  wvar.setTitle(title)

template window*(wvar : untyped, title : string) =
  var wvar : Window
  wvar.setTitle(title)

proc get_available_backends*() : seq[Backend] =
  result = @[]
  var bout = af_getAvailableBackends()
  

  for i in @[Backend.UNIFIED,Backend.CPU,Backend.CUDA,Backend.OPENCL] :
    if (bout and i.ord) != 0:
      result.add(i)  

proc set_backend_preferred*(preferred:seq[Backend] = 
  @[Backend.OPENCL, Backend.CUDA, Backend.CPU]) : Backend =
  let backends=get_available_backends()

  for b in preferred:    
    if b in backends:
      set_backend(b)
      result=b
      break


proc to_seq_typed[S,T](a : Matrix, count: int, s: typedesc[S], t: typedesc[T] ) : seq[T] =
  result=newSeq[T]()
  let dtype=a.dtype
  let c_item_size = get_DType_size(dtype)
  let num_items= if count>0: count else: len(a)

  let cdata : pointer = alloc0(c_item_size*len(a))
  a.host(cdata)

  for i in 0..<num_items:
    when S is Complex32:
      var real_ptr=cast[ptr float32](cast[int](cdata) + (i * c_item_size))    
      var imag_ptr=cast[ptr float32](cast[int](cdata) + (i * c_item_size)+4)    
      let source_item = complex32(real_ptr[],imag_ptr[])
    elif S is Complex64:
      var real_ptr=cast[ptr float64](cast[int](cdata) + (i * c_item_size))    
      var imag_ptr=cast[ptr float64](cast[int](cdata) + (i * c_item_size)+8)    
      let source_item = complex64(real_ptr[],imag_ptr[])
    else:
      var c_ptr=cast[ptr S](cast[int](cdata) + (i * c_item_size))    
      let source_item=c_ptr[]
    
    when T is Complex32:
      result.add(complex32(source_item))
    elif T is Complex64:
      result.add(complex64(source_item))
    else:
      result.add(T(source_item))

  dealloc(cdata)

proc to_seq*[T](m: Matrix, t: typedesc[T], count: int = -1) : seq[T] =
  ##[
  Get the all elements of a matrix as a sequence of type T defined
  ]##
  case m.dtype
    of DType.f32 : to_seq_typed(m,count,float32,T)
    of DType.c32 : to_seq_typed(m,count,float32,T)
    of DType.f64 : to_seq_typed(m,count,float64,T)
    of DType.c64 : to_seq_typed(m,count,float64,T)
    of DType.b8  : to_seq_typed(m,count,bool,T)
    of DType.s32 : to_seq_typed(m,count,int32,T)
    of DType.u32 : to_seq_typed(m,count,uint32,T)
    of DType.u8  : to_seq_typed(m,count,uint8,T)
    of DType.s64 : to_seq_typed(m,count,int64,T)
    of DType.u64 : to_seq_typed(m,count,uint64,T)
    of DType.s16 : to_seq_typed(m,count,int16,T)
    of DType.u16 : to_seq_typed(m,count,uint16,T)

proc first_as*[T](m: Matrix, t: typedesc[T]) : T =
  m.to_seq(t,1)[0]