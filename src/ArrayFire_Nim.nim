import strutils
import tables
import times
import typetraits
import std/sequtils
import std/algorithm
import std/math
include ArrayFire_Nim/raw


when sizeof(int) == 8:
   const sysint* = DType.S64
   const sysuint* = DType.U64
else:
   const sysint* = DType.S32
   const sysuint* = DType.U32

when sizeof(float) == 8:
   const sysfloat* = DType.F64
else:
   const sysfloat = DType.F32

type
   Complex32* {.final, header: "arrayfire.h", importcpp: "af::cfloat".} = object
      real*: cfloat
      imag*: cfloat

   Complex64* {.final, header: "arrayfire.h",
         importcpp: "af::cdouble".} = object
      real*: cdouble
      imag*: cdouble

type
   BatchFuncT* = proc (lhs: AFArray; rhs: AFArray): AFArray {.cdecl.}

type
   Cell* = object
      row*: cint
      col*: cint
      title*: cstring
      cmap*: Colormap

   Dim1* = tuple
      d0: int

   Dim2* = tuple
      d0: int
      d1: int

   Dim3* = tuple
      d0: int
      d1: int
      d2: int


proc dim4s*[T: int | DimT](dims: openarray[T]): Dim4 =
   var all_dims = [DimT(1), DimT(1), DimT(1), DimT(1)]
   let count = min(4, len(dims))
   for i in 0..<count:
      all_dims[i] = Dimt(dims[i])
   dim4(all_dims[0], all_dims[1], all_dims[2], all_dims[3])

proc dim4*[T: int | DimT](dims: varargs[T]): Dim4 =
   dim4s(dims)

converter toDimT*(i: int): DimT = DimT(i) ##automatically convert a single int to a DimT

proc `[]`*(this: var Dim4; dim: cuint): var DimT
  {.importcpp: "#[@]", header: "arrayfire.h".}

proc `[]`*(this: Dim4; dim: cuint): DimT
  {.noSideEffect, cdecl, importcpp: "#[@]", header: "arrayfire.h".}

proc `[]`*(d: Dim4; i: int): int =
   ##Index access to the dim4 dimensions
   int(d[cuint(i)])

proc `$`*(d: Dim4): string =
   var elems = newSeq[DimT]()
   for i in 0..<4:
      elems.add(DimT(d[i]))
   "Dim4[$1]"%join(elems, ", ")

proc toString*(a: AFArray; exp: string; precision: cint = 4; transpose: bool = true): string =
   let cexp = cstring(exp)
   let cresult = toString(cexp, a, precision, transpose)
   result = $(cresult)

proc `$`*(a: AFArray): string = toString(a, "")

proc `$`*(c: Complex32): string = "CF($1 + $2i)" % [$c.real, $c.imag]
proc `$`*(c: Complex64): string = "CD($1 + $2i)" % [$c.real, $c.imag]


proc complex32*[R, I](r: R; i: I): Complex32 = Complex32(real: cfloat(r),
      imag: cfloat(i))
proc complex64*[R, I](r: R; i: I): Complex64 = Complex64(real: cdouble(r),
      imag: cdouble(i))

proc complex32*[R](r: R): Complex32 =
   when R is Complex32:
      r
   elif R is Complex64:
      Complex32(r.real, r.imag)
   else:
      Complex32(real: cfloat(r), imag: cfloat(0))

proc complex64*[R](r: R): Complex64 =
   when R is Complex64:
      r
   elif R is Complex32:
      Complex64(r.real, r.imag)
   else:
      Complex64(real: cdouble(r), imag: cdouble(0))

# ----------------- Syntactic Sugar -----------------

const dtype_map = {
  "int": sysint,
  "uint": sysuint,
  "float": sysfloat,
  "Complex32": Dtype.C32,
  "Complex64": Dtype.C64,
  "float32": Dtype.F32,
  "float64": Dtype.F64,
  "int16": Dtype.S16,
  "int32": Dtype.S32,
  "int64": Dtype.S64,
  "uint8": Dtype.U8,
  "uint16": Dtype.U16,
  "uint32": Dtype.U32,
  "uint64": Dtype.U64,
  "bool": Dtype.B8
   }.toTable()

proc get_DType*(tname: string): DType =
   result = dtype_map[tname]

const dtype_size_map = {
  DType.F32: sizeof(float32),
  DType.C32: sizeof(float32)*2,
  DType.F64: sizeof(float64),
  DType.C64: sizeof(float64)*2,
  DType.B8: sizeof(bool),
  DType.S32: sizeof(int32),
  DType.U32: sizeof(uint32),
  DType.U8: sizeof(uint8),
  DType.S64: sizeof(int64),
  DType.U64: sizeof(uint64),
  DType.S16: sizeof(int16),
  DType.U16: sizeof(uint16),
}.toTable()


proc get_DType_size*(at: DType): int =
   result = dtype_size_map[at]

let iend*: cint = -1

proc get_AFArray*(this: AF_SEQ): AFArray
  {.importcpp: "#.operator af::array()", header: "arrayfire.h".}

converter toAFArray*(s: AF_SEQ): AFArray = s.get_AFArray()

converter toCuint*(d: DimT): cuint = cuint(d)

converter toInt*(i: cint): int = int(i)

when sizeof(clong) != sizeof(cint):
   converter toInt*(i: clong): int = int(i)

#region Construct Empty Arrays

proc afa*(): AFArray
  {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}

proc afa*(handle: AF_Array_Handle): AFArray
  {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}

proc afa*(ain: AFArray): AFArray
  {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}

proc afa*(dims: Dim4; ty: Dtype): AFArray
  {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}

proc afa*(dim0: DimT; ty: Dtype = Dtype.F32): AFArray =
   afa(dims = dim4(dim0), ty = ty)

proc afa*(dim0: DimT; dim1: DimT; ty: Dtype = Dtype.F32): AFArray =
   afa(dim4(dim0, dim1), ty)

proc afa*(dim0: DimT; dim1: DimT; dim2: DimT; ty: Dtype = Dtype.F32): AFArray =
   afa(dim4(dim0, dim1, dim2), ty)

proc afa*(dim0: DimT; dim1: DimT; dim2: DimT; dim3: DimT;
      ty: Dtype = Dtype.F32): AFArray =
   afa(dim4(dim0, dim1, dim2, dim3), ty)

proc afa*(input: AFArray; dims: Dim4): AFArray
  {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}

proc afa*(input: AFArray; dim0: DimT; dim1: DimT = 1; dim2: DimT = 1;
      dim3: DimT = 1): AFArray
  {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}

#endregion

#region template constructor calls

proc afa*[T](dim0: DimT; p0: ptr T; src: Source = Source.AFHOST): AFArray
  {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}

proc afa*[T](dim0: DimT; dim1: DimT; p0: ptr T, src: Source = Source.AFHOST): AFArray
  {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}

proc afa*[T](dim0: DimT; dim1: DimT; dim2: DimT; p0: ptr T, src: Source = Source.AFHOST): AFArray
  {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}

proc afa*[T](dim0: DimT; dim1: DimT; dim2: DimT; dim3: DimT; p0: ptr T, src: Source = Source.AFHOST): AFArray
  {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}

proc afa*[T](dims: Dim4; p0: ptr T; src: Source = Source.AFHOST): AFArray
  {.constructor importcpp: "af::array(@)", header: "arrayfire.h".}

#endregion


#region array construction from nim types

proc afa*[T](dims: Dim4, data: openarray[T]): AFArray = 
   when(T is int):  # arrayfire doesn't instantiate the template for int64
      var ns =  newSeq[int32]()
      for i in data:
         ns.add(int32(i))
      let cdata : ptr int32 = ns[0].addr
   else:
      var ns = data.toSeq()
      let cdata : ptr T = ns[0].addr
   result = afa(dims, cdata, src=Source.AFHOST)

proc afa*[T](dims: Dim4, data: Slice[T], afDtype: Dtype=Dtype.S32): AFArray =
   result = afa(dims, data.toSeq())
   if result.dtype() != afDtype:
      result = result.af_as(afDtype)
   

proc afa*[T](dims: Dim4; data: openarray[T]; afDtype: DType): AFArray =
   result = afa(dims, data).af_as(afDtype)

#endregion


#region array construction convenience procs

proc afa*[T](dim0: DimT; data: openarray[T]): AFArray =
   result = afa(dim4(dim0), data)

proc afa*[T](dim0: DimT; dim1: DimT; data: openarray[T]): AFArray =
   result = afa(dim4(dim0, dim1), data)

proc afa*[T](dim0: DimT; dim1: DimT; dim2: DimT; data: openarray[T]): AFArray =
   result = afa(dim4(dim0, dim1, dim2), data)

proc afa*[T](dim0: DimT; dim1: DimT; dim2: DimT; dim3: DimT; data: openarray[T]): AFArray =
   result = afa[T](dim4(dim0, dim1, dim2, dim3), data)

proc afa*[T](dim0: DimT; data: openarray[T]; afDtype: DType): AFArray =
   result = afa(dim4(dim0), data, afDtype)

proc afa*[T](dim0: DimT; dim1: DimT; data: openarray[T], afDtype: DType): AFArray =
   result = afa(dim4(dim0, dim1), data, afDtype)

proc afa*[T](dim0: DimT; dim1: DimT; dim2: DimT; data: openarray[T], afDtype: DType): AFArray =
   result = afa[T](dim4(dim0, dim1, dim2), data, afDtype)

proc afa*[T](dim0: DimT; dim1: DimT; dim2: DimT; dim3: DimT; data: openarray[T], afDtype: DType): AFArray =
   result = afa(dim4(dim0, dim1, dim2, dim3), data, afDtype)

proc afa*[T](data: openArray[T]; afDtype: Dtype = Dtype.S32): AFArray =
   result = afa(dim4(len(data), 1, 1, 1), data, afDtype)

proc afa*[T](dim0: DimT; data: Slice[T], afDtype: Dtype = Dtype.S32): AFArray =
   result = afa(dim4(dim0), data, afDtype)

proc afa*[T](dim0: DimT; dim1: DimT; data: Slice[T], afDtype: Dtype = Dtype.S32): AFArray =
   result = afa(dim4(dim0, dim1), data, afDtype)

proc afa*[T](dim0: DimT; dim1: DimT; dim2: DimT; data: Slice[T], afDtype: Dtype = Dtype.S32): AFArray =
   result = afa(dim4(dim0, dim1, dim2), data, afDtype)

proc afa*[T](dim0: DimT; dim1: DimT; dim2: DimT; dim3: DimT; data: Slice[T], afDtype: Dtype = Dtype.S32): AFArray =
   result = afa(dim4(dim0, dim1, dim2, dim3), data, afDtype)

#endregion

proc ndims*(m: AFArray): DimT =
   result = m.dims.ndims

proc randu*(dims: openarray[int]; ty: Dtype = Dtype.F32): AFArray =
   result = randu(dim4s(dims), ty)

proc randn*(dims: openarray[int]; ty: Dtype = Dtype.F32): AFArray =
   result = randn(dim4s(dims), ty)

proc aseq*[T1: int | float | int32 | int64](last: T1): AF_Seq =
   result = af_seq(cdouble(last))

proc aseq*[T1: int | float | int32 | int64;
           T2: int | float | int32 | int64](first: T1; last: T2): AF_Seq =
   result = af_seq(cdouble(first), cdouble(last), 1)

proc aseq*[T1: int | float | int32 | int64;
           T2: int | float | int32 | int64;
           T3: int | float | int32 | int64; ](first: T1; last: T2;
                 step: T3): AF_Seq =
   result = af_seq(cdouble(first), cdouble(last), cdouble(step))

proc aseq*(afs: AF_SEQ; isGfor: bool): AF_Seq 
  {.cdecl, constructor,  importcpp: "af::seq(@)", header : "arrayfire.h".}

proc begin(this: AF_Seq): cdouble
  {.importcpp: "#.s.begin", header: "arrayfire.h".}

proc until(this: AF_Seq): cdouble
  {.importcpp: "#.s.end", header: "arrayfire.h".}

proc step(this: AF_Seq): cdouble
  {.importcpp: "#.s.step", header: "arrayfire.h".}

proc getContent*(m: AF_Seq): tuple[start: float; until: float; step: float] =
   (m.begin, m.until, m.step)

proc `$`*(m: AF_Seq): string =
   let vals = m.getContent
   "AF_Seq[from: $1, until $2, step: $3]"%[$vals[0], $vals[1], $vals[2]]

proc `[]`*[I: int | int64 | AF_Seq | AFArray | IndexT;
           M: AFArray](m: M; i: I): AFArray
  {.noinit importcpp: "#(@)", header: "arrayfire.h".}

proc `[]`*[I1: int | int64 | AF_Seq | AFArray | IndexT;
          I2: int | int64 | AF_Seq | AFArray | IndexT;
          M: AFArray ](m: M; i1: I1; i2: I2): AFArray
  {.importcpp: "#(@)", header: "arrayfire.h".}

proc `[]`*[I1: int | int64 | AF_Seq | AFArray | IndexT;
           I2: int | int64 | AF_Seq | AFArray | IndexT;
           I3: int | int64 | AF_Seq | AFArray | IndexT;
           M: AFArray 
   ](m: M; idx1: I1; idx2: I2; idx3: I3): AFArray =
  {.importcpp: "#(#)", header: "arrayfire.h".}


proc `[]`*[I1: int | int64 | AF_Seq | AFArray | IndexT;
           I2: int | int64 | AF_Seq | AFArray | IndexT;
           I3: int | int64 | AF_Seq | AFArray | IndexT;
           I4: int | int64 | AF_Seq | AFArray | IndexT;
           M: AFArray 
   ](m: M; idx1: I1; idx2: I2; idx3: I3; idx4: I4): AFArray =
  {.importcpp: "#(@)", header: "arrayfire.h".}


proc `[]=`*[I1: int | int64 | AF_Seq | IndexT | AFArray;
            V: cdouble | cfloat | cint | cuint | clong | culong | clonglong | culonglong | char | bool | AFArray ;
            M: AFArray 
   ](this: var M; idx1: I1; val: V)
  {.importcpp: "#(#) = @", header: "arrayfire.h".}


proc `[]=`*[I1: int | int64 | AF_Seq | IndexT | AFArray;
             I2: int | int64 | AF_Seq | IndexT | AFArray;
             V: cdouble | cfloat | cint | cuint | clong | culong | clonglong |
               culonglong | char | bool | AFArray;
             M: AFArray
   ](this: var M; idx1: I1; idx2: I2; val: V)
  {.importcpp: "#(#,#) = @", header: "arrayfire.h".}


proc `[]=`*[I1: int | int64 | AF_Seq | IndexT | AFArray;
             I2: int | int64 | AF_Seq | IndexT | AFArray;
             I3: int | int64 | AF_Seq | IndexT | AFArray;
             V: cdouble | cfloat | cint | cuint | clong | culong | clonglong |
               culonglong | char | bool | AFArray;
             M: AFArray 
   ](this: var M; idx1: I1; idx2: I2; idx3: I3; val: V)
  {.importcpp: "#(#,#,#) = @", header: "arrayfire.h".}


proc `[]=`*[I1: int | int64 | AF_Seq | IndexT | AFArray;
             I2: int | int64 | AF_Seq | IndexT | AFArray;
             I3: int | int64 | AF_Seq | IndexT | AFArray;
             I4: int | int64 | AF_Seq | IndexT | AFArray;
             V: cdouble | cfloat | cint | cuint | clong | culong | clonglong |
               culonglong | char | bool | AFArray;
             M: AFArray
   ](this: var M; idx1: I1; idx2: I2; idx3: I3; idx4: I4; val: V)
  {.importcpp: "#(#,#,#,#) = @", header: "arrayfire.h".}


proc len*(m: AFArray): int =
   int(m.elements())

proc af_timeit*(fn: proc ()): cdouble {.importcpp: "af::timeit(@)", header: "arrayfire.h".}


template timeit_nim*(actions: untyped) : untyped =
   let t0 = getTime()
   actions
   let duration = getTime() - t0
   float(inNanoseconds(duration)) / float(1e9)

type Timer {.final, header : "arrayfire.h", importcpp: "af::timer".} = object
proc timer_start*(): Timer {.importcpp: "af::timer::start(@)", header: "arrayfire.h".}
proc timer_stop*(start: Timer): cdouble  {.importcpp: "af::timer::stop(@)", header: "arrayfire.h".}


template timeit*(actions: untyped) : untyped =
   const targetDurationPerTest = 0.050'f64
   const testSamples = 2 
   const minCycles = 3 
   var cycles = minCycles
   const nrSamples = 10

   var x = newSeq[float64]()

   for s in -testSamples..nrSamples:
      sync(-1)

      let timer = timer_start()

      for i in 0..cycles:
         actions
      sync(-1)

      let t = timer_stop(timer)
      if s >= 0:
         x.add(t)
      else:
         cycles = max(minCycles, int(trunc(targetDurationPerTest / t * float(cycles))))
   sort(x, SortOrder.Ascending)
   x[int(nrSamples / 2)] / float(cycles)

   

template gfor*(s, last, actions: untyped) =
   var s = aseq(aseq(cdouble(last)), true)
   while(gforToggle()):
      actions

template gfor*(s, first, last, actions: untyped) =
   s = aseq(aseq(cdouble(first), cdouble(last)), true)
   while(gforToggle()):
      actions

template gfor*(s, first, last, step, actions: untyped) =
   s = aseq(aseq(cdouble(first), cdouble(last), cdouble(step)), true)
   while(gforToggle()):
      actions

template window*(wvar: untyped; width: int; height: int; title: string) =
   var wvar: Window
   wvar.setSize(width, height)
   wvar.setTitle(title)

template window*(wvar: untyped; title: string) =
   var wvar: Window
   wvar.setTitle(title)

proc to_seq_typed[S, T](a: AFArray; count: int; s: typedesc[S]; t: typedesc[ T]): seq[T] =
  result = newSeq[T]()
  let dtype = a.dtype()
  let c_item_size = get_DType_size(dtype)
  let num_items = if count > 0: count else: len(a)

  let cdata: pointer = alloc0(c_item_size*len(a))
  a.host(cdata)

  for i in 0..<num_items:
    when S is Complex32:
        var real_ptr = cast[ptr float32](cast[int](cdata) + (i * c_item_size))
        var imag_ptr = cast[ptr float32](cast[int](cdata) + (i *
              c_item_size)+4)
        let source_item = complex32(real_ptr[], imag_ptr[])
    elif S is Complex64:
        var real_ptr = cast[ptr float64](cast[int](cdata) + (i * c_item_size))
        var imag_ptr = cast[ptr float64](cast[int](cdata) + (i *
              c_item_size)+8)
        let source_item = complex64(real_ptr[], imag_ptr[])
    else:
        var c_ptr = cast[ptr S](cast[int](cdata) + (i * c_item_size))
        let source_item = c_ptr[]

    when T is Complex32:
        result.add(complex32(source_item))
    elif T is Complex64:
        result.add(complex64(source_item))
    elif (T is int32 or T is int64 or T is int) and (S is float32 or S is float64 or S is float):
      result.add(T(round(source_item)))
    else:
      result.add(T(source_item))

  dealloc(cdata)

proc to_seq*[T](m: AFArray; t: typedesc[T]; count: int = -1): seq[T] =
   ##[
  Get the all elements of a AFArray as a sequence of type T defined
  ]##
   case m.dtype()
      of DType.F16: to_seq_typed(m, count, float32, T)
      of DType.F32: to_seq_typed(m, count, float32, T)
      of DType.C32: to_seq_typed(m, count, float32, T)
      of DType.F64: to_seq_typed(m, count, float64, T)
      of DType.C64: to_seq_typed(m, count, float64, T)
      of DType.B8: to_seq_typed(m, count, bool, T)
      of DType.S32: to_seq_typed(m, count, int32, T)
      of DType.U32: to_seq_typed(m, count, uint32, T)
      of DType.U8: to_seq_typed(m, count, uint8, T)
      of DType.S64: to_seq_typed(m, count, int64, T)
      of DType.U64: to_seq_typed(m, count, uint64, T)
      of DType.S16: to_seq_typed(m, count, int16, T)
      of DType.U16: to_seq_typed(m, count, uint16, T)

proc first_as*[T](m: AFArray; t: typedesc[T]): T =
   m.to_seq(t, 1)[0]

proc available_backends*(): seq[Backend] =
   let backends = getAvailableBackends()
   if (backends and ord(Backend.AF_BACKEND_CPU)) != 0:
      result.add(Backend.AF_BACKEND_CPU)
   if (backends and ord(Backend.AF_BACKEND_CUDA)) != 0:
      result.add(Backend.AF_BACKEND_CUDA)
   if (backends and ord(Backend.AF_BACKEND_OPENCL)) != 0:
      result.add(Backend.AF_BACKEND_OPENCL)

let span* = index(aseq(1, 1, 0))

proc constant*[T](val: T; dims: Dim4; ty: Dtype ): AFArray 
  {.cdecl,importcpp: "af::constant(@)", header : "arrayfire.h".}

proc constant*[T](val: T; dims: Dim4): AFArray 
  {.cdecl,importcpp: "af::constant(@)", header : "arrayfire.h".}

proc constant*[T](val: T; d0: DimT; ty: Dtype ): AFArray 
  {.cdecl,importcpp: "af::constant(@)", header : "arrayfire.h".}

proc constant*[T](val: T; d0: DimT): AFArray 
  {.cdecl,importcpp: "af::constant(@)", header : "arrayfire.h".}

proc constant*[T](val: T; d0: DimT; d1: DimT; ty: Dtype): AFArray 
  {.cdecl, importcpp: "af::constant(@)", header : "arrayfire.h".}

proc constant*[T](val: T; d0: DimT; d1: DimT; d2: DimT; ty: Dtype): AFArray 
  {.cdecl, importcpp: "af::constant(@)", header : "arrayfire.h".}

proc constant*[T](val: T; d0: DimT; d1: DimT; d2: DimT): AFArray 
  {.cdecl, importcpp: "af::constant(@)", header : "arrayfire.h".}

proc constant*[T](val: T; d0: DimT; d1: DimT; d2: DimT; d3: DimT;ty: Dtype = Dtype.F32 ): AFArray 
  {.cdecl,importcpp: "af::constant(@)", header : "arrayfire.h".}

proc row*(this: var AFArray; index: cint): AFArray 
  {.cdecl, importcpp: "row", header : "arrayfire.h".}

proc rows*(this: var AFArray; first: cint; last: cint): AFArray
  {.cdecl, importcpp: "rows", header : "arrayfire.h".}
  
proc col*(this: var AFArray; index: cint): AFArray
  {.cdecl, importcpp: "col", header : "arrayfire.h".}

proc cols*(this: var AFArray; first: cint; last: cint): AFArray 
  {.cdecl, importcpp: "cols", header : "arrayfire.h".}

template gfor*(s, last, actions: untyped) =
  var s = aseq(aseq(cdouble(last)),true)
  while(gforToggle()):
    actions

template gfor*(s, first, last, actions: untyped) =
  s = aseq(aseq(cdouble(first),cdouble(last)),true)
  while(gforToggle()):
    actions

template gfor*(s, first, last, step, actions: untyped) =
  s = aseq(aseq(cdouble(first),cdouble(last),cdouble(step)),true) 
  while(gforToggle()):
    actions

proc af_sum_all(real : ptr[cdouble], imag : ptr[cdouble], carray : AF_Array_Handle) : Err 
    {.importcpp: "af_sum_all(@)",header: "arrayfire.h".}

proc s_native_sum*(af_in: AFArray) : tuple[real: float, imag: float] =
  var real : cdouble = 0
  var imag : cdouble = 0
  discard af_sum_all(addr real, addr imag, af_in.get())
  (real,imag)

proc sum_as_int*(af_in: AFArray) : int =
  int(s_native_sum(af_in)[0])

proc sum_as_float*(af_in: AFArray) : float =
  float(s_native_sum(af_in)[0])

proc sum_as_complex*(af_in: AFArray) : Complex64 =
  var (real,imag) = s_native_sum(af_in)
  complex64(real,imag)

proc asum*(af_in: AFArray) : AFArray = asum(af_in, -1)

proc matmul*(lhs: AFArray; rhs: AFArray): AFArray = matmul(lhs, rhs, MatProp.AF_MAT_NONE, MatProp.AF_MAT_NONE)

proc fft*(af_in: AFArray) : AFArray = fft(af_in, 0)

proc fft2*(af_in: AFArray): AFArray = fft2(af_in, 0, 0)

proc grad*(af_in: AFArray) : tuple[dx: AFArray, dy : AFArray] =
  var dx : AFArray
  var dy : AFArray 

  grad(dx, dy, af_in)
  (dx, dy)

proc `[]`*(this: var Window; r: cint; c: cint): var Window 
  {.cdecl, importcpp: "#(@)", header : "arrayfire.h".}

proc tile*(af_in : AFArray; x: cuint; y: cuint = 1; z: cuint = 1; w: cuint = 1): AFArray 
  {.cdecl, importcpp: "af::tile(@)", header : "arrayfire.h".}

proc tile*(af_in : AFArray; dims: Dim4): AFArray 
  {.cdecl, importcpp: "af::tile(@)", header : "arrayfire.h".}

proc scalar*[T](this: AFArray): T 
  {.noSideEffect, cdecl, importcpp: "#.scalar<'*0>()", header : "arrayfire.h".}

proc scalar_r*(this: AFArray): cdouble 
  {.noSideEffect, cdecl, importcpp: "#.scalar<double>()", header : "arrayfire.h".}

proc svd*(afin: AFArray): (AFArray, AFArray, AFArray) =
   let u = afa()
   let s = afa()  
   let v = afa() 
   svd(u, s, v, afin)
   return (u, s, v)
