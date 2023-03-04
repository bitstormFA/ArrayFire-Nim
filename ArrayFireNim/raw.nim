when defined(Windows): 
  from os import nil 
  const AF_INCLUDE_PATH = "\"" & os.joinPath(os.getEnv("AF_PATH"), "include")  & "\""
  const AF_LIB_PATH =  "\"" & os.joinPath(os.getEnv("AF_PATH"), "lib") & "\""
  {.passC: "-D __FUNCSIG__ -std=c++11" & " -I" & AF_INCLUDE_PATH.}
  {.passL: "-lopengl32 -laf" & " -L" & AF_LIB_PATH.}
elif defined(Linux):
  {.passC: "-std=c++11".}
  {.passL: "-lGL -laf".}
elif defined(MacOsX):
  from os import nil
  const AF_INCLUDE_PATH = "\"" & os.joinPath(os.getEnv("AF_PATH"), "include") & "\""
  const AF_LIB_PATH = "\"" & os.joinPath(os.getEnv("AF_PATH"), "lib") & "\""
  {.passC: "-std=c++11" & " -I " & AF_INCLUDE_PATH.}
  {.passL: "-laf" & " -L" & AF_LIB_PATH.}
when sizeof(int) == 8:
  type DimT* = clonglong
else:
  type DimT* = cint  

type
  AF_Array_Handle* = distinct pointer

#region Types 

type
  AF_Seq* {.final, header : "arrayfire.h", importcpp: "af::seq".} = object
  Dim4* {.final, header : "arrayfire.h", importcpp: "af::dim4".} = object
  AF_Exception* {.final, header : "arrayfire.h", importcpp: "af::exception".} = object
  IndexT* {.final, header : "arrayfire.h", importcpp: "af::index".} = object
  AFArray* {.final, header : "arrayfire.h", importcpp: "af::array".} = object
  Event* {.final, header : "arrayfire.h", importcpp: "af::event".} = object
  Features* {.final, header : "arrayfire.h", importcpp: "af::features".} = object
  Window* {.final, header : "arrayfire.h", importcpp: "af::Window".} = object
  RandomEngine* {.final, header : "arrayfire.h", importcpp: "af::randomEngine".} = object

#endregion

#region Enums

type
  Err* {.pure, header : "arrayfire.h", import_cpp: "af_err", size: sizeof(cint).} = enum 
    AF_SUCCESS = 0,
    AF_ERR_NO_MEM = 101,
    AF_ERR_DRIVER = 102,
    AF_ERR_RUNTIME = 103,
    AF_ERR_INVALID_ARRAY = 201,
    AF_ERR_ARG = 202,
    AF_ERR_SIZE = 203,
    AF_ERR_TYPE = 204,
    AF_ERR_DIFF_TYPE = 205,
    AF_ERR_BATCH = 207,
    AF_ERR_DEVICE = 208,
    AF_ERR_NOT_SUPPORTED = 301,
    AF_ERR_NOT_CONFIGURED = 302,
    AF_ERR_NONFREE = 303,
    AF_ERR_NO_DBL = 401,
    AF_ERR_NO_GFX = 402,
    AF_ERR_NO_HALF = 403,
    AF_ERR_LOAD_LIB = 501,
    AF_ERR_LOAD_SYM = 502,
    AF_ERR_ARR_BKND_MISMATCH = 503,
    AF_ERR_INTERNAL = 998,
    AF_ERR_UNKNOWN = 999

  Dtype* {.pure, header : "arrayfire.h", import_cpp: "af_dtype", size: sizeof(cint).} = enum 
    F32 = 0,
    C32 = 1,
    F64 = 2,
    C64 = 3,
    B8 = 4,
    S32 = 5,
    U32 = 6,
    U8 = 7,
    S64 = 8,
    U64 = 9,
    S16 = 10,
    U16 = 11,
    F16 = 12

  Source* {.pure, header : "arrayfire.h", import_cpp: "af_source", size: sizeof(cint).} = enum 
    AFDEVICE = 0,
    AFHOST = 1

  InterpType* {.pure, header : "arrayfire.h", import_cpp: "af_interp_type", size: sizeof(cint).} = enum 
    AF_INTERP_NEAREST = 0,
    AF_INTERP_LINEAR = 1,
    AF_INTERP_BILINEAR = 2,
    AF_INTERP_CUBIC = 3,
    AF_INTERP_LOWER = 4,
    AF_INTERP_LINEAR_COSINE = 5,
    AF_INTERP_BILINEAR_COSINE = 6,
    AF_INTERP_BICUBIC = 7,
    AF_INTERP_CUBIC_SPLINE = 8,
    AF_INTERP_BICUBIC_SPLINE = 9

  BorderType* {.pure, header : "arrayfire.h", import_cpp: "af_border_type", size: sizeof(cint).} = enum 
    AF_PAD_ZERO = 0,
    AF_PAD_SYM = 1,
    AF_PAD_CLAMP_TO_EDGE = 2,
    AF_PAD_PERIODIC = 3

  Connectivity* {.pure, header : "arrayfire.h", import_cpp: "af_connectivity", size: sizeof(cint).} = enum 
    AF_CONNECTIVITY_4 = 4,
    AF_CONNECTIVITY_8 = 8

  ConvMode* {.pure, header : "arrayfire.h", import_cpp: "af_conv_mode", size: sizeof(cint).} = enum 
    AF_CONV_DEFAULT = 0,
    AF_CONV_EXPAND = 1

  ConvDomain* {.pure, header : "arrayfire.h", import_cpp: "af_conv_domain", size: sizeof(cint).} = enum 
    AF_CONV_AUTO = 0,
    AF_CONV_SPATIAL = 1,
    AF_CONV_FREQ = 2

  MatchType* {.pure, header : "arrayfire.h", import_cpp: "af_match_type", size: sizeof(cint).} = enum 
    AF_SAD = 0,
    AF_ZSAD = 1,
    AF_LSAD = 2,
    AF_SSD = 3,
    AF_ZSSD = 4,
    AF_LSSD = 5,
    AF_NCC = 6,
    AF_ZNCC = 7,
    AF_SHD = 8

  YccStd* {.pure, header : "arrayfire.h", import_cpp: "af_ycc_std", size: sizeof(cint).} = enum 
    AF_YCC_601 = 601,
    AF_YCC_709 = 709,
    AF_YCC_2020 = 2020

  CspaceT* {.pure, header : "arrayfire.h", import_cpp: "af_cspace_t", size: sizeof(cint).} = enum 
    AF_GRAY = 0,
    AF_RGB = 1,
    AF_HSV = 2,
    AF_YCBCR = 3

  MatProp* {.pure, header : "arrayfire.h", import_cpp: "af_mat_prop", size: sizeof(cint).} = enum 
    AF_MAT_NONE = 0,
    AF_MAT_TRANS = 1,
    AF_MAT_CTRANS = 2,
    AF_MAT_CONJ = 4,
    AF_MAT_UPPER = 32,
    AF_MAT_LOWER = 64,
    AF_MAT_DIAG_UNIT = 128,
    AF_MAT_SYM = 512,
    AF_MAT_POSDEF = 1024,
    AF_MAT_ORTHOG = 2048,
    AF_MAT_TRI_DIAG = 4096,
    AF_MAT_BLOCK_DIAG = 8192

  NormType* {.pure, header : "arrayfire.h", import_cpp: "af_norm_type", size: sizeof(cint).} = enum 
    AF_NORM_VECTOR_1 = 0,
    AF_NORM_VECTOR_INF = 1,
    AF_NORM_EUCLID = 2,
    AF_NORM_VECTOR_P = 3,
    AF_NORM_MATRIX_1 = 4,
    AF_NORM_MATRIX_INF = 5,
    AF_NORM_MATRIX_2 = 6,
    AF_NORM_MATRIX_L_PQ = 7

  ImageFormat* {.pure, header : "arrayfire.h", import_cpp: "af_image_format", size: sizeof(cint).} = enum 
    AF_FIF_BMP = 0,
    AF_FIF_ICO = 1,
    AF_FIF_JPEG = 2,
    AF_FIF_JNG = 3,
    AF_FIF_PNG = 13,
    AF_FIF_PPM = 14,
    AF_FIF_PPMRAW = 15,
    AF_FIF_TIFF = 18,
    AF_FIF_PSD = 20,
    AF_FIF_HDR = 26,
    AF_FIF_EXR = 29,
    AF_FIF_JP2 = 31,
    AF_FIF_RAW = 34

  MomentType* {.pure, header : "arrayfire.h", import_cpp: "af_moment_type", size: sizeof(cint).} = enum 
    AF_MOMENT_M00 = 1,
    AF_MOMENT_M01 = 2,
    AF_MOMENT_M10 = 4,
    AF_MOMENT_M11 = 8,
    AF_MOMENT_FIRST_ORDER = 15

  HomographyType* {.pure, header : "arrayfire.h", import_cpp: "af_homography_type", size: sizeof(cint).} = enum 
    AF_HOMOGRAPHY_RANSAC = 0,
    AF_HOMOGRAPHY_LMEDS = 1

  Backend* {.pure, header : "arrayfire.h", import_cpp: "af_backend", size: sizeof(cint).} = enum 
    AF_BACKEND_DEFAULT = 0,
    AF_BACKEND_CPU = 1,
    AF_BACKEND_CUDA = 2,
    AF_BACKEND_OPENCL = 4

  SomeenumT* {.pure, header : "arrayfire.h", import_cpp: "SomeenumT", size: sizeof(cint).} = enum 
    AF_ID = 0

  BinaryOp* {.pure, header : "arrayfire.h", import_cpp: "af_binary_op", size: sizeof(cint).} = enum 
    AF_BINARY_ADD = 0,
    AF_BINARY_MUL = 1,
    AF_BINARY_MIN = 2,
    AF_BINARY_MAX = 3

  RandomEngineType* {.pure, header : "arrayfire.h", import_cpp: "af_random_engine_type", size: sizeof(cint).} = enum 
    AF_RANDOM_ENGINE_PHILOX = 100,
    AF_RANDOM_ENGINE_THREEFRY = 200,
    AF_RANDOM_ENGINE_MERSENNE = 300

  Colormap* {.pure, header : "arrayfire.h", import_cpp: "af_colormap", size: sizeof(cint).} = enum 
    AF_COLORMAP_DEFAULT = 0,
    AF_COLORMAP_SPECTRUM = 1,
    AF_COLORMAP_COLORS = 2,
    AF_COLORMAP_RED = 3,
    AF_COLORMAP_MOOD = 4,
    AF_COLORMAP_HEAT = 5,
    AF_COLORMAP_BLUE = 6,
    AF_COLORMAP_INFERNO = 7,
    AF_COLORMAP_MAGMA = 8,
    AF_COLORMAP_PLASMA = 9,
    AF_COLORMAP_VIRIDIS = 10

  MarkerType* {.pure, header : "arrayfire.h", import_cpp: "af_marker_type", size: sizeof(cint).} = enum 
    AF_MARKER_NONE = 0,
    AF_MARKER_POINT = 1,
    AF_MARKER_CIRCLE = 2,
    AF_MARKER_SQUARE = 3,
    AF_MARKER_TRIANGLE = 4,
    AF_MARKER_CROSS = 5,
    AF_MARKER_PLUS = 6,
    AF_MARKER_STAR = 7

  CannyThreshold* {.pure, header : "arrayfire.h", import_cpp: "af_canny_threshold", size: sizeof(cint).} = enum 
    AF_CANNY_THRESHOLD_MANUAL = 0,
    AF_CANNY_THRESHOLD_AUTO_OTSU = 1

  Storage* {.pure, header : "arrayfire.h", import_cpp: "af_storage", size: sizeof(cint).} = enum 
    AF_STORAGE_DENSE = 0,
    AF_STORAGE_CSR = 1,
    AF_STORAGE_CSC = 2,
    AF_STORAGE_COO = 3

  FluxFunction* {.pure, header : "arrayfire.h", import_cpp: "af_flux_function", size: sizeof(cint).} = enum 
    AF_FLUX_DEFAULT = 0,
    AF_FLUX_QUADRATIC = 1,
    AF_FLUX_EXPONENTIAL = 2

  DiffusionEq* {.pure, header : "arrayfire.h", import_cpp: "af_diffusion_eq", size: sizeof(cint).} = enum 
    AF_DIFFUSION_DEFAULT = 0,
    AF_DIFFUSION_GRAD = 1,
    AF_DIFFUSION_MCDE = 2

  TopkFunction* {.pure, header : "arrayfire.h", import_cpp: "af_topk_function", size: sizeof(cint).} = enum 
    AF_TOPK_DEFAULT = 0,
    AF_TOPK_MIN = 1,
    AF_TOPK_MAX = 2

  VarBias* {.pure, header : "arrayfire.h", import_cpp: "af_var_bias", size: sizeof(cint).} = enum 
    AF_VARIANCE_DEFAULT = 0,
    AF_VARIANCE_SAMPLE = 1,
    AF_VARIANCE_POPULATION = 2

  IterativeDeconvAlgo* {.pure, header : "arrayfire.h", import_cpp: "af_iterative_deconv_algo", size: sizeof(cint).} = enum 
    AF_ITERATIVE_DECONV_DEFAULT = 0,
    AF_ITERATIVE_DECONV_LANDWEBER = 1,
    AF_ITERATIVE_DECONV_RICHARDSONLUCY = 2

  InverseDeconvAlgo* {.pure, header : "arrayfire.h", import_cpp: "af_inverse_deconv_algo", size: sizeof(cint).} = enum 
    AF_INVERSE_DECONV_DEFAULT = 0,
    AF_INVERSE_DECONV_TIKHONOV = 1

  ConvGradientType* {.pure, header : "arrayfire.h", import_cpp: "af_conv_gradient_type", size: sizeof(cint).} = enum 
    AF_CONV_GRADIENT_DEFAULT = 0,
    AF_CONV_GRADIENT_FILTER = 1,
    AF_CONV_GRADIENT_DATA = 2,
    AF_CONV_GRADIENT_BIAS = 3

#endregion


#region Functions

proc devicecount*(  ) : int32 {.importcpp: "af::devicecount(@)", header: "arrayfire.h".}
proc deviceget*(  ) : int32 {.importcpp: "af::deviceget(@)", header: "arrayfire.h".}
proc deviceset*( device : int32 )  {.importcpp: "af::deviceset(@)", header: "arrayfire.h".}
proc loadimage*( filename : cstring, is_color : bool ) : AFArray {.importcpp: "af::loadimage(@)", header: "arrayfire.h".}
proc saveimage*( filename : cstring, af_in : AFArray )  {.importcpp: "af::saveimage(@)", header: "arrayfire.h".}
proc gaussiankernel*( rows : int32, cols : int32, sig_r : float64, sig_c : float64 ) : AFArray {.importcpp: "af::gaussiankernel(@)", header: "arrayfire.h".}
proc alltrue*( af_in : AFArray, dim : int32 ) : AFArray {.importcpp: "af::alltrue(@)", header: "arrayfire.h".}
proc anytrue*( af_in : AFArray, dim : int32 ) : AFArray {.importcpp: "af::anytrue(@)", header: "arrayfire.h".}
proc setunique*( af_in : AFArray, is_sorted : bool ) : AFArray {.importcpp: "af::setunique(@)", header: "arrayfire.h".}
proc setunion*( first : AFArray, second : AFArray, is_unique : bool ) : AFArray {.importcpp: "af::setunion(@)", header: "arrayfire.h".}
proc setintersect*( first : AFArray, second : AFArray, is_unique : bool ) : AFArray {.importcpp: "af::setintersect(@)", header: "arrayfire.h".}
proc histequal*( af_in : AFArray, hist : AFArray ) : AFArray {.importcpp: "af::histequal(@)", header: "arrayfire.h".}
proc colorspace*( image : AFArray, to : CSpaceT, af_from : CSpaceT ) : AFArray {.importcpp: "af::colorspace(@)", header: "arrayfire.h".}
proc filter*( image : AFArray, kernel : AFArray ) : AFArray {.importcpp: "af::filter(@)", header: "arrayfire.h".}
proc mul*( af_in : AFArray, dim : int32 ) : AFArray {.importcpp: "af::mul(@)", header: "arrayfire.h".}
proc deviceprop*( d_name : cstring, d_platform : cstring, d_toolkit : cstring, d_compute : cstring )  {.importcpp: "af::deviceprop(@)", header: "arrayfire.h".}
proc asum*( af_in : AFArray, dim : int32 ) : AFArray {.importcpp: "af::sum(@)", header: "arrayfire.h".}
proc asum*( af_in : AFArray, dim : int32, nanval : float64 ) : AFArray {.importcpp: "af::sum(@)", header: "arrayfire.h".}
proc sumByKey*( keys_out : AFArray, vals_out : AFArray, keys : AFArray, vals : AFArray, dim : int32 )  {.importcpp: "af::sumByKey(@)", header: "arrayfire.h".}
proc sumByKey*( keys_out : AFArray, vals_out : AFArray, keys : AFArray, vals : AFArray, dim : int32, nanval : float64 )  {.importcpp: "af::sumByKey(@)", header: "arrayfire.h".}
proc product*( af_in : AFArray, dim : int32 ) : AFArray {.importcpp: "af::product(@)", header: "arrayfire.h".}
proc product*( af_in : AFArray, dim : int32, nanval : float64 ) : AFArray {.importcpp: "af::product(@)", header: "arrayfire.h".}
proc productByKey*( keys_out : AFArray, vals_out : AFArray, keys : AFArray, vals : AFArray, dim : int32 )  {.importcpp: "af::productByKey(@)", header: "arrayfire.h".}
proc productByKey*( keys_out : AFArray, vals_out : AFArray, keys : AFArray, vals : AFArray, dim : int32, nanval : float64 )  {.importcpp: "af::productByKey(@)", header: "arrayfire.h".}
proc amin*( af_in : AFArray, dim : int32 ) : AFArray {.importcpp: "af::min(@)", header: "arrayfire.h".}
proc minByKey*( keys_out : AFArray, vals_out : AFArray, keys : AFArray, vals : AFArray, dim : int32 )  {.importcpp: "af::minByKey(@)", header: "arrayfire.h".}
proc amax*( af_in : AFArray, dim : int32 ) : AFArray {.importcpp: "af::max(@)", header: "arrayfire.h".}
proc maxByKey*( keys_out : AFArray, vals_out : AFArray, keys : AFArray, vals : AFArray, dim : int32 )  {.importcpp: "af::maxByKey(@)", header: "arrayfire.h".}
proc amax*( val : AFArray, idx : AFArray, af_in : AFArray, ragged_len : AFArray, dim : int32 )  {.importcpp: "af::max(@)", header: "arrayfire.h".}
proc allTrueByKey*( keys_out : AFArray, vals_out : AFArray, keys : AFArray, vals : AFArray, dim : int32 )  {.importcpp: "af::allTrueByKey(@)", header: "arrayfire.h".}
proc anyTrueByKey*( keys_out : AFArray, vals_out : AFArray, keys : AFArray, vals : AFArray, dim : int32 )  {.importcpp: "af::anyTrueByKey(@)", header: "arrayfire.h".}
proc count*( af_in : AFArray, dim : int32 ) : AFArray {.importcpp: "af::count(@)", header: "arrayfire.h".}
proc countByKey*( keys_out : AFArray, vals_out : AFArray, keys : AFArray, vals : AFArray, dim : int32 )  {.importcpp: "af::countByKey(@)", header: "arrayfire.h".}
proc amin*( val : AFArray, idx : AFArray, af_in : AFArray, dim : int32 )  {.importcpp: "af::min(@)", header: "arrayfire.h".}
proc amax*( val : AFArray, idx : AFArray, af_in : AFArray, dim : int32 )  {.importcpp: "af::max(@)", header: "arrayfire.h".}
proc accum*( af_in : AFArray, dim : int32 ) : AFArray {.importcpp: "af::accum(@)", header: "arrayfire.h".}
proc scan*( af_in : AFArray, dim : int32, op : BinaryOp, inclusive_scan : bool ) : AFArray {.importcpp: "af::scan(@)", header: "arrayfire.h".}
proc scanByKey*( key : AFArray, af_in : AFArray, dim : int32, op : BinaryOp, inclusive_scan : bool ) : AFArray {.importcpp: "af::scanByKey(@)", header: "arrayfire.h".}
proc where*( af_in : AFArray ) : AFArray {.importcpp: "af::where(@)", header: "arrayfire.h".}
proc diff1*( af_in : AFArray, dim : int32 ) : AFArray {.importcpp: "af::diff1(@)", header: "arrayfire.h".}
proc diff2*( af_in : AFArray, dim : int32 ) : AFArray {.importcpp: "af::diff2(@)", header: "arrayfire.h".}
proc sort*( af_in : AFArray, dim : uint32, isAscending : bool ) : AFArray {.importcpp: "af::sort(@)", header: "arrayfire.h".}
proc sort*( af_out : AFArray, indices : AFArray, af_in : AFArray, dim : uint32, isAscending : bool )  {.importcpp: "af::sort(@)", header: "arrayfire.h".}
proc sort*( out_keys : AFArray, out_values : AFArray, keys : AFArray, values : AFArray, dim : uint32, isAscending : bool )  {.importcpp: "af::sort(@)", header: "arrayfire.h".}
proc amin*( lhs : AFArray, rhs : AFArray ) : AFArray {.importcpp: "af::min(@)", header: "arrayfire.h".}
proc amin*( lhs : AFArray, rhs : float64 ) : AFArray {.importcpp: "af::min(@)", header: "arrayfire.h".}
proc amin*( lhs : float64, rhs : AFArray ) : AFArray {.importcpp: "af::min(@)", header: "arrayfire.h".}
proc amax*( lhs : AFArray, rhs : AFArray ) : AFArray {.importcpp: "af::max(@)", header: "arrayfire.h".}
proc amax*( lhs : AFArray, rhs : float64 ) : AFArray {.importcpp: "af::max(@)", header: "arrayfire.h".}
proc amax*( lhs : float64, rhs : AFArray ) : AFArray {.importcpp: "af::max(@)", header: "arrayfire.h".}
proc clamp*( af_in : AFArray, lo : AFArray, hi : AFArray ) : AFArray {.importcpp: "af::clamp(@)", header: "arrayfire.h".}
proc clamp*( af_in : AFArray, lo : AFArray, hi : float64 ) : AFArray {.importcpp: "af::clamp(@)", header: "arrayfire.h".}
proc clamp*( af_in : AFArray, lo : float64, hi : AFArray ) : AFArray {.importcpp: "af::clamp(@)", header: "arrayfire.h".}
proc clamp*( af_in : AFArray, lo : float64, hi : float64 ) : AFArray {.importcpp: "af::clamp(@)", header: "arrayfire.h".}
proc rem*( lhs : AFArray, rhs : AFArray ) : AFArray {.importcpp: "af::rem(@)", header: "arrayfire.h".}
proc rem*( lhs : AFArray, rhs : float64 ) : AFArray {.importcpp: "af::rem(@)", header: "arrayfire.h".}
proc rem*( lhs : float64, rhs : AFArray ) : AFArray {.importcpp: "af::rem(@)", header: "arrayfire.h".}
proc af_mod*( lhs : AFArray, rhs : AFArray ) : AFArray {.importcpp: "af::mod(@)", header: "arrayfire.h".}
proc af_mod*( lhs : AFArray, rhs : float64 ) : AFArray {.importcpp: "af::mod(@)", header: "arrayfire.h".}
proc af_mod*( lhs : float64, rhs : AFArray ) : AFArray {.importcpp: "af::mod(@)", header: "arrayfire.h".}
proc abs*( af_in : AFArray ) : AFArray {.importcpp: "af::abs(@)", header: "arrayfire.h".}
proc arg*( af_in : AFArray ) : AFArray {.importcpp: "af::arg(@)", header: "arrayfire.h".}
proc sign*( af_in : AFArray ) : AFArray {.importcpp: "af::sign(@)", header: "arrayfire.h".}
proc round*( af_in : AFArray ) : AFArray {.importcpp: "af::round(@)", header: "arrayfire.h".}
proc trunc*( af_in : AFArray ) : AFArray {.importcpp: "af::trunc(@)", header: "arrayfire.h".}
proc floor*( af_in : AFArray ) : AFArray {.importcpp: "af::floor(@)", header: "arrayfire.h".}
proc ceil*( af_in : AFArray ) : AFArray {.importcpp: "af::ceil(@)", header: "arrayfire.h".}
proc hypot*( lhs : AFArray, rhs : AFArray ) : AFArray {.importcpp: "af::hypot(@)", header: "arrayfire.h".}
proc hypot*( lhs : AFArray, rhs : float64 ) : AFArray {.importcpp: "af::hypot(@)", header: "arrayfire.h".}
proc hypot*( lhs : float64, rhs : AFArray ) : AFArray {.importcpp: "af::hypot(@)", header: "arrayfire.h".}
proc sin*( af_in : AFArray ) : AFArray {.importcpp: "af::sin(@)", header: "arrayfire.h".}
proc cos*( af_in : AFArray ) : AFArray {.importcpp: "af::cos(@)", header: "arrayfire.h".}
proc tan*( af_in : AFArray ) : AFArray {.importcpp: "af::tan(@)", header: "arrayfire.h".}
proc asin*( af_in : AFArray ) : AFArray {.importcpp: "af::asin(@)", header: "arrayfire.h".}
proc acos*( af_in : AFArray ) : AFArray {.importcpp: "af::acos(@)", header: "arrayfire.h".}
proc atan*( af_in : AFArray ) : AFArray {.importcpp: "af::atan(@)", header: "arrayfire.h".}
proc atan2*( lhs : AFArray, rhs : AFArray ) : AFArray {.importcpp: "af::atan2(@)", header: "arrayfire.h".}
proc atan2*( lhs : AFArray, rhs : float64 ) : AFArray {.importcpp: "af::atan2(@)", header: "arrayfire.h".}
proc atan2*( lhs : float64, rhs : AFArray ) : AFArray {.importcpp: "af::atan2(@)", header: "arrayfire.h".}
proc sinh*( af_in : AFArray ) : AFArray {.importcpp: "af::sinh(@)", header: "arrayfire.h".}
proc cosh*( af_in : AFArray ) : AFArray {.importcpp: "af::cosh(@)", header: "arrayfire.h".}
proc tanh*( af_in : AFArray ) : AFArray {.importcpp: "af::tanh(@)", header: "arrayfire.h".}
proc asinh*( af_in : AFArray ) : AFArray {.importcpp: "af::asinh(@)", header: "arrayfire.h".}
proc acosh*( af_in : AFArray ) : AFArray {.importcpp: "af::acosh(@)", header: "arrayfire.h".}
proc atanh*( af_in : AFArray ) : AFArray {.importcpp: "af::atanh(@)", header: "arrayfire.h".}
proc complex*( af_in : AFArray ) : AFArray {.importcpp: "af::complex(@)", header: "arrayfire.h".}
proc complex*( real : AFArray, imag : AFArray ) : AFArray {.importcpp: "af::complex(@)", header: "arrayfire.h".}
proc complex*( real : AFArray, imag : float64 ) : AFArray {.importcpp: "af::complex(@)", header: "arrayfire.h".}
proc complex*( real : float64, imag : AFArray ) : AFArray {.importcpp: "af::complex(@)", header: "arrayfire.h".}
proc real*( af_in : AFArray ) : AFArray {.importcpp: "af::real(@)", header: "arrayfire.h".}
proc imag*( af_in : AFArray ) : AFArray {.importcpp: "af::imag(@)", header: "arrayfire.h".}
proc conjg*( af_in : AFArray ) : AFArray {.importcpp: "af::conjg(@)", header: "arrayfire.h".}
proc root*( nth_root : AFArray, value : AFArray ) : AFArray {.importcpp: "af::root(@)", header: "arrayfire.h".}
proc root*( nth_root : AFArray, value : float64 ) : AFArray {.importcpp: "af::root(@)", header: "arrayfire.h".}
proc root*( nth_root : float64, value : AFArray ) : AFArray {.importcpp: "af::root(@)", header: "arrayfire.h".}
proc pow*( base : AFArray, exponent : AFArray ) : AFArray {.importcpp: "af::pow(@)", header: "arrayfire.h".}
proc pow*( base : AFArray, exponent : float64 ) : AFArray {.importcpp: "af::pow(@)", header: "arrayfire.h".}
proc pow*( base : float64, exponent : AFArray ) : AFArray {.importcpp: "af::pow(@)", header: "arrayfire.h".}
proc pow2*( af_in : AFArray ) : AFArray {.importcpp: "af::pow2(@)", header: "arrayfire.h".}
proc sigmoid*( af_in : AFArray ) : AFArray {.importcpp: "af::sigmoid(@)", header: "arrayfire.h".}
proc exp*( af_in : AFArray ) : AFArray {.importcpp: "af::exp(@)", header: "arrayfire.h".}
proc expm1*( af_in : AFArray ) : AFArray {.importcpp: "af::expm1(@)", header: "arrayfire.h".}
proc erf*( af_in : AFArray ) : AFArray {.importcpp: "af::erf(@)", header: "arrayfire.h".}
proc erfc*( af_in : AFArray ) : AFArray {.importcpp: "af::erfc(@)", header: "arrayfire.h".}
proc log*( af_in : AFArray ) : AFArray {.importcpp: "af::log(@)", header: "arrayfire.h".}
proc log1p*( af_in : AFArray ) : AFArray {.importcpp: "af::log1p(@)", header: "arrayfire.h".}
proc log10*( af_in : AFArray ) : AFArray {.importcpp: "af::log10(@)", header: "arrayfire.h".}
proc log2*( af_in : AFArray ) : AFArray {.importcpp: "af::log2(@)", header: "arrayfire.h".}
proc sqrt*( af_in : AFArray ) : AFArray {.importcpp: "af::sqrt(@)", header: "arrayfire.h".}
proc rsqrt*( af_in : AFArray ) : AFArray {.importcpp: "af::rsqrt(@)", header: "arrayfire.h".}
proc cbrt*( af_in : AFArray ) : AFArray {.importcpp: "af::cbrt(@)", header: "arrayfire.h".}
proc factorial*( af_in : AFArray ) : AFArray {.importcpp: "af::factorial(@)", header: "arrayfire.h".}
proc tgamma*( af_in : AFArray ) : AFArray {.importcpp: "af::tgamma(@)", header: "arrayfire.h".}
proc lgamma*( af_in : AFArray ) : AFArray {.importcpp: "af::lgamma(@)", header: "arrayfire.h".}
proc iszero*( af_in : AFArray ) : AFArray {.importcpp: "af::iszero(@)", header: "arrayfire.h".}
proc isInf*( af_in : AFArray ) : AFArray {.importcpp: "af::isInf(@)", header: "arrayfire.h".}
proc isNaN*( af_in : AFArray ) : AFArray {.importcpp: "af::isNaN(@)", header: "arrayfire.h".}
proc info*(  )  {.importcpp: "af::info(@)", header: "arrayfire.h".}
proc infoString*( verbose : bool ) : cstring {.importcpp: "af::infoString(@)", header: "arrayfire.h".}
proc deviceInfo*( d_name : cstring, d_platform : cstring, d_toolkit : cstring, d_compute : cstring )  {.importcpp: "af::deviceInfo(@)", header: "arrayfire.h".}
proc getDeviceCount*(  ) : int32 {.importcpp: "af::getDeviceCount(@)", header: "arrayfire.h".}
proc getDevice*(  ) : int32 {.importcpp: "af::getDevice(@)", header: "arrayfire.h".}
proc isDoubleAvailable*( device : int32 ) : bool {.importcpp: "af::isDoubleAvailable(@)", header: "arrayfire.h".}
proc isHalfAvailable*( device : int32 ) : bool {.importcpp: "af::isHalfAvailable(@)", header: "arrayfire.h".}
proc setDevice*( device : int32 )  {.importcpp: "af::setDevice(@)", header: "arrayfire.h".}
proc sync*( device : int32 )  {.importcpp: "af::sync(@)", header: "arrayfire.h".}
proc af_alloc*( elements : uint32, af_type : Dtype )  {.importcpp: "af::alloc(@)", header: "arrayfire.h".}
proc allocV2*( bytes : uint32 )  {.importcpp: "af::allocV2(@)", header: "arrayfire.h".}
proc free*( af_ptr : pointer )  {.importcpp: "af::free(@)", header: "arrayfire.h".}
proc freeV2*( af_ptr : pointer )  {.importcpp: "af::freeV2(@)", header: "arrayfire.h".}
proc pinned*( elements : uint32, af_type : Dtype )  {.importcpp: "af::pinned(@)", header: "arrayfire.h".}
proc freePinned*( af_ptr : pointer )  {.importcpp: "af::freePinned(@)", header: "arrayfire.h".}
proc allocHost*( elements : uint32, af_type : Dtype )  {.importcpp: "af::allocHost(@)", header: "arrayfire.h".}
proc freeHost*( af_ptr : pointer )  {.importcpp: "af::freeHost(@)", header: "arrayfire.h".}
proc deviceMemInfo*( alloc_bytes : uint32, alloc_buffers : uint32, lock_bytes : uint32, lock_buffers : uint32 )  {.importcpp: "af::deviceMemInfo(@)", header: "arrayfire.h".}
proc printMemInfo*( msg : cstring, device_id : int32 )  {.importcpp: "af::printMemInfo(@)", header: "arrayfire.h".}
proc deviceGC*(  )  {.importcpp: "af::deviceGC(@)", header: "arrayfire.h".}
proc setMemStepSize*( size : uint32 )  {.importcpp: "af::setMemStepSize(@)", header: "arrayfire.h".}
proc getMemStepSize*(  ) : uint32 {.importcpp: "af::getMemStepSize(@)", header: "arrayfire.h".}
proc `+`*(first : Dim4, second : Dim4) : Dim4 {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `-`*(first : Dim4, second : Dim4) : Dim4 {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `*`*(first : Dim4, second : Dim4) : Dim4 {.importcpp: "(# * #)", header: "arrayfire.h".}
proc isSpan*( af_seq : AF_Seq ) : bool {.importcpp: "af::isSpan(@)", header: "arrayfire.h".}
proc seqElements*( af_seq : AF_Seq ) : uint32 {.importcpp: "af::seqElements(@)", header: "arrayfire.h".}
proc calcDim*( af_seq : AF_Seq, parentDim : DimT ) : DimT {.importcpp: "af::calcDim(@)", header: "arrayfire.h".}
proc lookup*( af_in : AFArray, idx : AFArray, dim : int32 ) : AFArray {.importcpp: "af::lookup(@)", header: "arrayfire.h".}
proc copy*( dst : AFArray, src : AFArray, idx0 : IndexT, idx1 : IndexT, idx2 : IndexT, idx3 : IndexT )  {.importcpp: "af::copy(@)", header: "arrayfire.h".}
proc print*( exp : cstring, arr : AFArray )  {.importcpp: "af::print(@)", header: "arrayfire.h".}
proc print*( exp : cstring, arr : AFArray, precision : int32 )  {.importcpp: "af::print(@)", header: "arrayfire.h".}
proc saveArray*( key : cstring, arr : AFArray, filename : cstring, append : bool ) : int32 {.importcpp: "af::saveArray(@)", header: "arrayfire.h".}
proc readArray*( filename : cstring, index : uint32 ) : AFArray {.importcpp: "af::readArray(@)", header: "arrayfire.h".}
proc readArray*( filename : cstring, key : cstring ) : AFArray {.importcpp: "af::readArray(@)", header: "arrayfire.h".}
proc readArrayCheck*( filename : cstring, key : cstring ) : int32 {.importcpp: "af::readArrayCheck(@)", header: "arrayfire.h".}
proc toString*( output : cstring, exp : cstring, arr : AFArray, precision : int32, transpose : bool )  {.importcpp: "af::toString(@)", header: "arrayfire.h".}
proc toString*( exp : cstring, arr : AFArray, precision : int32, transpose : bool ) : cstring {.importcpp: "af::toString(@)", header: "arrayfire.h".}
proc exampleFunction*( af_in : AFArray, param : SomeenumT ) : AFArray {.importcpp: "af::exampleFunction(@)", header: "arrayfire.h".}
proc getSizeOf*( af_type : Dtype ) : uint32 {.importcpp: "af::getSizeOf(@)", header: "arrayfire.h".}
proc real*( val : float64 ) : float32 {.importcpp: "af::real(@)", header: "arrayfire.h".}
proc imag*( val : float64 ) : float32 {.importcpp: "af::imag(@)", header: "arrayfire.h".}
proc abs*( val : float32 ) : float32 {.importcpp: "af::abs(@)", header: "arrayfire.h".}
proc abs*( val : float64 ) : float64 {.importcpp: "af::abs(@)", header: "arrayfire.h".}
proc conj*( val : float32 ) : float32 {.importcpp: "af::conj(@)", header: "arrayfire.h".}
proc conj*( val : float64 ) : float64 {.importcpp: "af::conj(@)", header: "arrayfire.h".}
proc `+`*(lhs : AFArray, rhs : AFArray) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : bool, rhs : AFArray) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : int32, rhs : AFArray) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : uint32, rhs : AFArray) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : cstring, rhs : AFArray) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : int64, rhs : AFArray) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : uint64, rhs : AFArray) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : float64, rhs : AFArray) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : float32, rhs : AFArray) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : AFArray, rhs : bool) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : AFArray, rhs : int32) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : AFArray, rhs : uint32) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : AFArray, rhs : cstring) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : AFArray, rhs : int64) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : AFArray, rhs : uint64) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : AFArray, rhs : float64) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : AFArray, rhs : float32) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : int16, rhs : AFArray) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : uint16, rhs : AFArray) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : AFArray, rhs : int16) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `+`*(lhs : AFArray, rhs : uint16) : AFArray {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `-`*(lhs : AFArray, rhs : AFArray) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : bool, rhs : AFArray) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : int32, rhs : AFArray) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : uint32, rhs : AFArray) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : cstring, rhs : AFArray) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : int64, rhs : AFArray) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : uint64, rhs : AFArray) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : float64, rhs : AFArray) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : float32, rhs : AFArray) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : AFArray, rhs : bool) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : AFArray, rhs : int32) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : AFArray, rhs : uint32) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : AFArray, rhs : cstring) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : AFArray, rhs : int64) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : AFArray, rhs : uint64) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : AFArray, rhs : float64) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : AFArray, rhs : float32) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : int16, rhs : AFArray) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : uint16, rhs : AFArray) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : AFArray, rhs : int16) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `-`*(lhs : AFArray, rhs : uint16) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `*`*(lhs : AFArray, rhs : AFArray) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : bool, rhs : AFArray) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : int32, rhs : AFArray) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : uint32, rhs : AFArray) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : cstring, rhs : AFArray) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : int64, rhs : AFArray) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : uint64, rhs : AFArray) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : float64, rhs : AFArray) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : float32, rhs : AFArray) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : AFArray, rhs : bool) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : AFArray, rhs : int32) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : AFArray, rhs : uint32) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : AFArray, rhs : cstring) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : AFArray, rhs : int64) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : AFArray, rhs : uint64) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : AFArray, rhs : float64) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : AFArray, rhs : float32) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : int16, rhs : AFArray) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : uint16, rhs : AFArray) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : AFArray, rhs : int16) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `*`*(lhs : AFArray, rhs : uint16) : AFArray {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `/`*(lhs : AFArray, rhs : AFArray) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : bool, rhs : AFArray) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : int32, rhs : AFArray) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : uint32, rhs : AFArray) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : cstring, rhs : AFArray) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : int64, rhs : AFArray) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : uint64, rhs : AFArray) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : float64, rhs : AFArray) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : float32, rhs : AFArray) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : AFArray, rhs : bool) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : AFArray, rhs : int32) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : AFArray, rhs : uint32) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : AFArray, rhs : cstring) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : AFArray, rhs : int64) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : AFArray, rhs : uint64) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : AFArray, rhs : float64) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : AFArray, rhs : float32) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : int16, rhs : AFArray) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : uint16, rhs : AFArray) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : AFArray, rhs : int16) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `/`*(lhs : AFArray, rhs : uint16) : AFArray {.importcpp: "(# / #)", header: "arrayfire.h".}
proc `==`*(lhs : AFArray, rhs : AFArray) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : bool, rhs : AFArray) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : int32, rhs : AFArray) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : uint32, rhs : AFArray) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : cstring, rhs : AFArray) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : int64, rhs : AFArray) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : uint64, rhs : AFArray) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : float64, rhs : AFArray) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : float32, rhs : AFArray) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : AFArray, rhs : bool) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : AFArray, rhs : int32) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : AFArray, rhs : uint32) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : AFArray, rhs : cstring) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : AFArray, rhs : int64) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : AFArray, rhs : uint64) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : AFArray, rhs : float64) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : AFArray, rhs : float32) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : int16, rhs : AFArray) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : uint16, rhs : AFArray) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : AFArray, rhs : int16) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `==`*(lhs : AFArray, rhs : uint16) : AFArray {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `!=`*(lhs : AFArray, rhs : AFArray) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : bool, rhs : AFArray) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : int32, rhs : AFArray) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : uint32, rhs : AFArray) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : cstring, rhs : AFArray) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : int64, rhs : AFArray) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : uint64, rhs : AFArray) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : float64, rhs : AFArray) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : float32, rhs : AFArray) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : AFArray, rhs : bool) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : AFArray, rhs : int32) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : AFArray, rhs : uint32) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : AFArray, rhs : cstring) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : AFArray, rhs : int64) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : AFArray, rhs : uint64) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : AFArray, rhs : float64) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : AFArray, rhs : float32) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : int16, rhs : AFArray) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : uint16, rhs : AFArray) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : AFArray, rhs : int16) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `!=`*(lhs : AFArray, rhs : uint16) : AFArray {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `<`*(lhs : AFArray, rhs : AFArray) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : bool, rhs : AFArray) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : int32, rhs : AFArray) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : uint32, rhs : AFArray) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : cstring, rhs : AFArray) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : int64, rhs : AFArray) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : uint64, rhs : AFArray) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : float64, rhs : AFArray) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : float32, rhs : AFArray) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : AFArray, rhs : bool) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : AFArray, rhs : int32) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : AFArray, rhs : uint32) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : AFArray, rhs : cstring) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : AFArray, rhs : int64) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : AFArray, rhs : uint64) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : AFArray, rhs : float64) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : AFArray, rhs : float32) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : int16, rhs : AFArray) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : uint16, rhs : AFArray) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : AFArray, rhs : int16) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<`*(lhs : AFArray, rhs : uint16) : AFArray {.importcpp: "(# < #)", header: "arrayfire.h".}
proc `<=`*(lhs : AFArray, rhs : AFArray) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : bool, rhs : AFArray) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : int32, rhs : AFArray) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : uint32, rhs : AFArray) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : cstring, rhs : AFArray) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : int64, rhs : AFArray) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : uint64, rhs : AFArray) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : float64, rhs : AFArray) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : float32, rhs : AFArray) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : AFArray, rhs : bool) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : AFArray, rhs : int32) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : AFArray, rhs : uint32) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : AFArray, rhs : cstring) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : AFArray, rhs : int64) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : AFArray, rhs : uint64) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : AFArray, rhs : float64) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : AFArray, rhs : float32) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : int16, rhs : AFArray) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : uint16, rhs : AFArray) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : AFArray, rhs : int16) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `<=`*(lhs : AFArray, rhs : uint16) : AFArray {.importcpp: "(# <= #)", header: "arrayfire.h".}
proc `>`*(lhs : AFArray, rhs : AFArray) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : bool, rhs : AFArray) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : int32, rhs : AFArray) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : uint32, rhs : AFArray) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : cstring, rhs : AFArray) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : int64, rhs : AFArray) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : uint64, rhs : AFArray) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : float64, rhs : AFArray) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : float32, rhs : AFArray) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : AFArray, rhs : bool) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : AFArray, rhs : int32) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : AFArray, rhs : uint32) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : AFArray, rhs : cstring) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : AFArray, rhs : int64) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : AFArray, rhs : uint64) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : AFArray, rhs : float64) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : AFArray, rhs : float32) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : int16, rhs : AFArray) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : uint16, rhs : AFArray) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : AFArray, rhs : int16) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>`*(lhs : AFArray, rhs : uint16) : AFArray {.importcpp: "(# > #)", header: "arrayfire.h".}
proc `>=`*(lhs : AFArray, rhs : AFArray) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : bool, rhs : AFArray) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : int32, rhs : AFArray) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : uint32, rhs : AFArray) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : cstring, rhs : AFArray) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : int64, rhs : AFArray) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : uint64, rhs : AFArray) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : float64, rhs : AFArray) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : float32, rhs : AFArray) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : AFArray, rhs : bool) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : AFArray, rhs : int32) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : AFArray, rhs : uint32) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : AFArray, rhs : cstring) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : AFArray, rhs : int64) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : AFArray, rhs : uint64) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : AFArray, rhs : float64) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : AFArray, rhs : float32) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : int16, rhs : AFArray) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : uint16, rhs : AFArray) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : AFArray, rhs : int16) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `>=`*(lhs : AFArray, rhs : uint16) : AFArray {.importcpp: "(# >= #)", header: "arrayfire.h".}
proc `||`*(lhs : AFArray, rhs : AFArray) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : bool, rhs : AFArray) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : int32, rhs : AFArray) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : uint32, rhs : AFArray) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : cstring, rhs : AFArray) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : int64, rhs : AFArray) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : uint64, rhs : AFArray) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : float64, rhs : AFArray) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : float32, rhs : AFArray) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : AFArray, rhs : bool) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : AFArray, rhs : int32) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : AFArray, rhs : uint32) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : AFArray, rhs : cstring) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : AFArray, rhs : int64) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : AFArray, rhs : uint64) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : AFArray, rhs : float64) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : AFArray, rhs : float32) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : int16, rhs : AFArray) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : uint16, rhs : AFArray) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : AFArray, rhs : int16) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `||`*(lhs : AFArray, rhs : uint16) : AFArray {.importcpp: "(# || #)", header: "arrayfire.h".}
proc `%`*(lhs : AFArray, rhs : AFArray) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : bool, rhs : AFArray) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : int32, rhs : AFArray) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : uint32, rhs : AFArray) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : cstring, rhs : AFArray) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : int64, rhs : AFArray) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : uint64, rhs : AFArray) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : float64, rhs : AFArray) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : float32, rhs : AFArray) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : AFArray, rhs : bool) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : AFArray, rhs : int32) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : AFArray, rhs : uint32) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : AFArray, rhs : cstring) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : AFArray, rhs : int64) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : AFArray, rhs : uint64) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : AFArray, rhs : float64) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : AFArray, rhs : float32) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : int16, rhs : AFArray) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : uint16, rhs : AFArray) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : AFArray, rhs : int16) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `%`*(lhs : AFArray, rhs : uint16) : AFArray {.importcpp: "(# % #)", header: "arrayfire.h".}
proc `|`*(lhs : AFArray, rhs : AFArray) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : bool, rhs : AFArray) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : int32, rhs : AFArray) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : uint32, rhs : AFArray) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : cstring, rhs : AFArray) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : int64, rhs : AFArray) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : uint64, rhs : AFArray) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : float64, rhs : AFArray) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : float32, rhs : AFArray) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : AFArray, rhs : bool) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : AFArray, rhs : int32) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : AFArray, rhs : uint32) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : AFArray, rhs : cstring) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : AFArray, rhs : int64) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : AFArray, rhs : uint64) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : AFArray, rhs : float64) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : AFArray, rhs : float32) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : int16, rhs : AFArray) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : uint16, rhs : AFArray) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : AFArray, rhs : int16) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `|`*(lhs : AFArray, rhs : uint16) : AFArray {.importcpp: "(# | #)", header: "arrayfire.h".}
proc `^`*(lhs : AFArray, rhs : AFArray) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : bool, rhs : AFArray) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : int32, rhs : AFArray) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : uint32, rhs : AFArray) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : cstring, rhs : AFArray) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : int64, rhs : AFArray) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : uint64, rhs : AFArray) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : float64, rhs : AFArray) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : float32, rhs : AFArray) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : AFArray, rhs : bool) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : AFArray, rhs : int32) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : AFArray, rhs : uint32) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : AFArray, rhs : cstring) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : AFArray, rhs : int64) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : AFArray, rhs : uint64) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : AFArray, rhs : float64) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : AFArray, rhs : float32) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : int16, rhs : AFArray) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : uint16, rhs : AFArray) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : AFArray, rhs : int16) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `^`*(lhs : AFArray, rhs : uint16) : AFArray {.importcpp: "(# ^ #)", header: "arrayfire.h".}
proc `<<`*(lhs : AFArray, rhs : AFArray) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : bool, rhs : AFArray) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : int32, rhs : AFArray) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : uint32, rhs : AFArray) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : cstring, rhs : AFArray) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : int64, rhs : AFArray) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : uint64, rhs : AFArray) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : float64, rhs : AFArray) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : float32, rhs : AFArray) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : AFArray, rhs : bool) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : AFArray, rhs : int32) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : AFArray, rhs : uint32) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : AFArray, rhs : cstring) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : AFArray, rhs : int64) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : AFArray, rhs : uint64) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : AFArray, rhs : float64) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : AFArray, rhs : float32) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : int16, rhs : AFArray) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : uint16, rhs : AFArray) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : AFArray, rhs : int16) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `<<`*(lhs : AFArray, rhs : uint16) : AFArray {.importcpp: "(# << #)", header: "arrayfire.h".}
proc `>>`*(lhs : AFArray, rhs : AFArray) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : bool, rhs : AFArray) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : int32, rhs : AFArray) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : uint32, rhs : AFArray) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : cstring, rhs : AFArray) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : int64, rhs : AFArray) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : uint64, rhs : AFArray) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : float64, rhs : AFArray) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : float32, rhs : AFArray) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : AFArray, rhs : bool) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : AFArray, rhs : int32) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : AFArray, rhs : uint32) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : AFArray, rhs : cstring) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : AFArray, rhs : int64) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : AFArray, rhs : uint64) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : AFArray, rhs : float64) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : AFArray, rhs : float32) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : int16, rhs : AFArray) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : uint16, rhs : AFArray) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : AFArray, rhs : int16) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `>>`*(lhs : AFArray, rhs : uint16) : AFArray {.importcpp: "(# >> #)", header: "arrayfire.h".}
proc `&`*(lhs : AFArray, rhs : AFArray) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : AFArray, rhs : bool) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : AFArray, rhs : float64) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : AFArray, rhs : float32) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : AFArray, rhs : cstring) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : AFArray, rhs : int32) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : AFArray, rhs : int64) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : AFArray, rhs : int16) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : AFArray, rhs : uint64) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : AFArray, rhs : uint32) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : AFArray, rhs : uint16) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : bool, rhs : AFArray) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : float64, rhs : AFArray) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : float32, rhs : AFArray) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : cstring, rhs : AFArray) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : int32, rhs : AFArray) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : int64, rhs : AFArray) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : int16, rhs : AFArray) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : uint64, rhs : AFArray) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : uint32, rhs : AFArray) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&`*(lhs : uint16, rhs : AFArray) : AFArray {.importcpp: "(# & #)", header: "arrayfire.h".}
proc `&&`*(lhs : AFArray, rhs : AFArray) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : AFArray, rhs : bool) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : AFArray, rhs : float64) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : AFArray, rhs : float32) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : AFArray, rhs : cstring) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : AFArray, rhs : int32) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : AFArray, rhs : int64) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : AFArray, rhs : int16) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : AFArray, rhs : uint64) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : AFArray, rhs : uint32) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : AFArray, rhs : uint16) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : bool, rhs : AFArray) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : float64, rhs : AFArray) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : float32, rhs : AFArray) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : cstring, rhs : AFArray) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : int32, rhs : AFArray) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : int64, rhs : AFArray) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : int16, rhs : AFArray) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : uint64, rhs : AFArray) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : uint32, rhs : AFArray) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc `&&`*(lhs : uint16, rhs : AFArray) : AFArray {.importcpp: "(# && #)", header: "arrayfire.h".}
proc eval*( num : int32, arrays : AFArray )  {.importcpp: "af::eval(@)", header: "arrayfire.h".}
proc eval*( a : AFArray, b : AFArray )  {.importcpp: "af::eval(@)", header: "arrayfire.h".}
proc eval*( a : AFArray, b : AFArray, c : AFArray )  {.importcpp: "af::eval(@)", header: "arrayfire.h".}
proc eval*( a : AFArray, b : AFArray, c : AFArray, d : AFArray )  {.importcpp: "af::eval(@)", header: "arrayfire.h".}
proc eval*( a : AFArray, b : AFArray, c : AFArray, d : AFArray, e : AFArray )  {.importcpp: "af::eval(@)", header: "arrayfire.h".}
proc eval*( a : AFArray, b : AFArray, c : AFArray, d : AFArray, e : AFArray, f : AFArray )  {.importcpp: "af::eval(@)", header: "arrayfire.h".}
proc setManualEvalFlag*( flag : bool )  {.importcpp: "af::setManualEvalFlag(@)", header: "arrayfire.h".}
proc getManualEvalFlag*(  ) : bool {.importcpp: "af::getManualEvalFlag(@)", header: "arrayfire.h".}
proc setBackend*( bknd : Backend )  {.importcpp: "af::setBackend(@)", header: "arrayfire.h".}
proc getBackendCount*(  ) : uint32 {.importcpp: "af::getBackendCount(@)", header: "arrayfire.h".}
proc getAvailableBackends*(  ) : int32 {.importcpp: "af::getAvailableBackends(@)", header: "arrayfire.h".}
proc getBackendId*( af_in : AFArray ) : Backend {.importcpp: "af::getBackendId(@)", header: "arrayfire.h".}
proc getActiveBackend*(  ) : Backend {.importcpp: "af::getActiveBackend(@)", header: "arrayfire.h".}
proc getDeviceId*( af_in : AFArray ) : int32 {.importcpp: "af::getDeviceId(@)", header: "arrayfire.h".}
proc matmul*( lhs : AFArray, rhs : AFArray, optLhs : MatProp, optRhs : MatProp ) : AFArray {.importcpp: "af::matmul(@)", header: "arrayfire.h".}
proc matmulNT*( lhs : AFArray, rhs : AFArray ) : AFArray {.importcpp: "af::matmulNT(@)", header: "arrayfire.h".}
proc matmulTN*( lhs : AFArray, rhs : AFArray ) : AFArray {.importcpp: "af::matmulTN(@)", header: "arrayfire.h".}
proc matmulTT*( lhs : AFArray, rhs : AFArray ) : AFArray {.importcpp: "af::matmulTT(@)", header: "arrayfire.h".}
proc matmul*( a : AFArray, b : AFArray, c : AFArray ) : AFArray {.importcpp: "af::matmul(@)", header: "arrayfire.h".}
proc matmul*( a : AFArray, b : AFArray, c : AFArray, d : AFArray ) : AFArray {.importcpp: "af::matmul(@)", header: "arrayfire.h".}
proc dot*( lhs : AFArray, rhs : AFArray, optLhs : MatProp, optRhs : MatProp ) : AFArray {.importcpp: "af::dot(@)", header: "arrayfire.h".}
proc transpose*( af_in : AFArray, conjugate : bool ) : AFArray {.importcpp: "af::transpose(@)", header: "arrayfire.h".}
proc transposeInPlace*( af_in : AFArray, conjugate : bool )  {.importcpp: "af::transposeInPlace(@)", header: "arrayfire.h".}
proc identity*( dims : Dim4, ty : Dtype ) : AFArray {.importcpp: "af::identity(@)", header: "arrayfire.h".}
proc identity*( d0 : DimT, ty : Dtype ) : AFArray {.importcpp: "af::identity(@)", header: "arrayfire.h".}
proc identity*( d0 : DimT, d1 : DimT, ty : Dtype ) : AFArray {.importcpp: "af::identity(@)", header: "arrayfire.h".}
proc identity*( d0 : DimT, d1 : DimT, d2 : DimT, ty : Dtype ) : AFArray {.importcpp: "af::identity(@)", header: "arrayfire.h".}
proc identity*( d0 : DimT, d1 : DimT, d2 : DimT, d3 : DimT, ty : Dtype ) : AFArray {.importcpp: "af::identity(@)", header: "arrayfire.h".}
proc range*( dims : Dim4, seq_dim : int32, ty : Dtype ) : AFArray {.importcpp: "af::range(@)", header: "arrayfire.h".}
proc range*( d0 : DimT, d1 : DimT, d2 : DimT, d3 : DimT, seq_dim : int32, ty : Dtype ) : AFArray {.importcpp: "af::range(@)", header: "arrayfire.h".}
proc iota*( dims : Dim4, tile_dims : Dim4, ty : Dtype ) : AFArray {.importcpp: "af::iota(@)", header: "arrayfire.h".}
proc diag*( af_in : AFArray, num : int32, extract : bool ) : AFArray {.importcpp: "af::diag(@)", header: "arrayfire.h".}
proc join*( dim : int32, first : AFArray, second : AFArray ) : AFArray {.importcpp: "af::join(@)", header: "arrayfire.h".}
proc join*( dim : int32, first : AFArray, second : AFArray, third : AFArray ) : AFArray {.importcpp: "af::join(@)", header: "arrayfire.h".}
proc join*( dim : int32, first : AFArray, second : AFArray, third : AFArray, fourth : AFArray ) : AFArray {.importcpp: "af::join(@)", header: "arrayfire.h".}
proc reorder*( af_in : AFArray, x : uint32, y : uint32, z : uint32, w : uint32 ) : AFArray {.importcpp: "af::reorder(@)", header: "arrayfire.h".}
proc shift*( af_in : AFArray, x : int32, y : int32, z : int32, w : int32 ) : AFArray {.importcpp: "af::shift(@)", header: "arrayfire.h".}
proc moddims*( af_in : AFArray, dims : Dim4 ) : AFArray {.importcpp: "af::moddims(@)", header: "arrayfire.h".}
proc moddims*( af_in : AFArray, d0 : DimT, d1 : DimT, d2 : DimT, d3 : DimT ) : AFArray {.importcpp: "af::moddims(@)", header: "arrayfire.h".}
proc moddims*( af_in : AFArray, ndims : uint32, dims : DimT ) : AFArray {.importcpp: "af::moddims(@)", header: "arrayfire.h".}
proc flat*( af_in : AFArray ) : AFArray {.importcpp: "af::flat(@)", header: "arrayfire.h".}
proc flip*( af_in : AFArray, dim : uint32 ) : AFArray {.importcpp: "af::flip(@)", header: "arrayfire.h".}
proc lower*( af_in : AFArray, is_unit_diag : bool ) : AFArray {.importcpp: "af::lower(@)", header: "arrayfire.h".}
proc upper*( af_in : AFArray, is_unit_diag : bool ) : AFArray {.importcpp: "af::upper(@)", header: "arrayfire.h".}
proc select*( cond : AFArray, a : AFArray, b : AFArray ) : AFArray {.importcpp: "af::select(@)", header: "arrayfire.h".}
proc select*( cond : AFArray, a : AFArray, b : float64 ) : AFArray {.importcpp: "af::select(@)", header: "arrayfire.h".}
proc select*( cond : AFArray, a : float64, b : AFArray ) : AFArray {.importcpp: "af::select(@)", header: "arrayfire.h".}
proc replace*( a : AFArray, cond : AFArray, b : AFArray )  {.importcpp: "af::replace(@)", header: "arrayfire.h".}
proc replace*( a : AFArray, cond : AFArray, b : float64 )  {.importcpp: "af::replace(@)", header: "arrayfire.h".}
proc pad*( af_in : AFArray, beginPadding : Dim4, endPadding : Dim4, padFillType : BorderType ) : AFArray {.importcpp: "af::pad(@)", header: "arrayfire.h".}
proc gforToggle*(  ) : bool {.importcpp: "af::gforToggle(@)", header: "arrayfire.h".}
proc gforGet*(  ) : bool {.importcpp: "af::gforGet(@)", header: "arrayfire.h".}
proc gforSet*( val : bool )  {.importcpp: "af::gforSet(@)", header: "arrayfire.h".}
proc grad*( dx : AFArray, dy : AFArray, af_in : AFArray )  {.importcpp: "af::grad(@)", header: "arrayfire.h".}
proc loadImageMem*( af_ptr : pointer ) : AFArray {.importcpp: "af::loadImageMem(@)", header: "arrayfire.h".}
proc saveImageMem*( af_in : AFArray, format : ImageFormat )  {.importcpp: "af::saveImageMem(@)", header: "arrayfire.h".}
proc deleteImageMem*( af_ptr : pointer )  {.importcpp: "af::deleteImageMem(@)", header: "arrayfire.h".}
proc loadImageNative*( filename : cstring ) : AFArray {.importcpp: "af::loadImageNative(@)", header: "arrayfire.h".}
proc saveImageNative*( filename : cstring, af_in : AFArray )  {.importcpp: "af::saveImageNative(@)", header: "arrayfire.h".}
proc isImageIOAvailable*(  ) : bool {.importcpp: "af::isImageIOAvailable(@)", header: "arrayfire.h".}
proc resize*( af_in : AFArray, odim0 : DimT, odim1 : DimT, af_mathod : InterpType ) : AFArray {.importcpp: "af::resize(@)", header: "arrayfire.h".}
proc resize*( scale0 : float32, scale1 : float32, af_in : AFArray, af_mathod : InterpType ) : AFArray {.importcpp: "af::resize(@)", header: "arrayfire.h".}
proc resize*( scale : float32, af_in : AFArray, af_mathod : InterpType ) : AFArray {.importcpp: "af::resize(@)", header: "arrayfire.h".}
proc rotate*( af_in : AFArray, theta : float32, crop : bool, af_mathod : InterpType ) : AFArray {.importcpp: "af::rotate(@)", header: "arrayfire.h".}
proc transform*( af_in : AFArray, transform : AFArray, odim0 : DimT, odim1 : DimT, af_mathod : InterpType, inverse : bool ) : AFArray {.importcpp: "af::transform(@)", header: "arrayfire.h".}
proc transformCoordinates*( tf : AFArray, d0 : float32, d1 : float32 ) : AFArray {.importcpp: "af::transformCoordinates(@)", header: "arrayfire.h".}
proc translate*( af_in : AFArray, trans0 : float32, trans1 : float32, odim0 : DimT, odim1 : DimT, af_mathod : InterpType ) : AFArray {.importcpp: "af::translate(@)", header: "arrayfire.h".}
proc scale*( af_in : AFArray, scale0 : float32, scale1 : float32, odim0 : DimT, odim1 : DimT, af_mathod : InterpType ) : AFArray {.importcpp: "af::scale(@)", header: "arrayfire.h".}
proc skew*( af_in : AFArray, skew0 : float32, skew1 : float32, odim0 : DimT, odim1 : DimT, inverse : bool, af_mathod : InterpType ) : AFArray {.importcpp: "af::skew(@)", header: "arrayfire.h".}
proc bilateral*( af_in : AFArray, spatial_sigma : float32, chromatic_sigma : float32, is_color : bool ) : AFArray {.importcpp: "af::bilateral(@)", header: "arrayfire.h".}
proc histogram*( af_in : AFArray, nbins : uint32, minval : float64, maxval : float64 ) : AFArray {.importcpp: "af::histogram(@)", header: "arrayfire.h".}
proc histogram*( af_in : AFArray, nbins : uint32 ) : AFArray {.importcpp: "af::histogram(@)", header: "arrayfire.h".}
proc meanShift*( af_in : AFArray, spatial_sigma : float32, chromatic_sigma : float32, iter : uint32, is_color : bool ) : AFArray {.importcpp: "af::meanShift(@)", header: "arrayfire.h".}
proc minfilt*( af_in : AFArray, wind_length : DimT, wind_width : DimT, edge_pad : BorderType ) : AFArray {.importcpp: "af::minfilt(@)", header: "arrayfire.h".}
proc maxfilt*( af_in : AFArray, wind_length : DimT, wind_width : DimT, edge_pad : BorderType ) : AFArray {.importcpp: "af::maxfilt(@)", header: "arrayfire.h".}
proc dilate*( af_in : AFArray, mask : AFArray ) : AFArray {.importcpp: "af::dilate(@)", header: "arrayfire.h".}
proc dilate3*( af_in : AFArray, mask : AFArray ) : AFArray {.importcpp: "af::dilate3(@)", header: "arrayfire.h".}
proc erode*( af_in : AFArray, mask : AFArray ) : AFArray {.importcpp: "af::erode(@)", header: "arrayfire.h".}
proc erode3*( af_in : AFArray, mask : AFArray ) : AFArray {.importcpp: "af::erode3(@)", header: "arrayfire.h".}
proc regions*( af_in : AFArray, connectivity : Connectivity, af_type : Dtype ) : AFArray {.importcpp: "af::regions(@)", header: "arrayfire.h".}
proc sobel*( dx : AFArray, dy : AFArray, img : AFArray, ker_size : uint32 )  {.importcpp: "af::sobel(@)", header: "arrayfire.h".}
proc sobel*( img : AFArray, ker_size : uint32, isFast : bool ) : AFArray {.importcpp: "af::sobel(@)", header: "arrayfire.h".}
proc rgb2gray*( af_in : AFArray, rPercent : float32, gPercent : float32, bPercent : float32 ) : AFArray {.importcpp: "af::rgb2gray(@)", header: "arrayfire.h".}
proc gray2rgb*( af_in : AFArray, rFactor : float32, gFactor : float32, bFactor : float32 ) : AFArray {.importcpp: "af::gray2rgb(@)", header: "arrayfire.h".}
proc hsv2rgb*( af_in : AFArray ) : AFArray {.importcpp: "af::hsv2rgb(@)", header: "arrayfire.h".}
proc rgb2hsv*( af_in : AFArray ) : AFArray {.importcpp: "af::rgb2hsv(@)", header: "arrayfire.h".}
proc unwrap*( af_in : AFArray, wx : DimT, wy : DimT, sx : DimT, sy : DimT, px : DimT, py : DimT, is_column : bool ) : AFArray {.importcpp: "af::unwrap(@)", header: "arrayfire.h".}
proc wrap*( af_in : AFArray, ox : DimT, oy : DimT, wx : DimT, wy : DimT, sx : DimT, sy : DimT, px : DimT, py : DimT, is_column : bool ) : AFArray {.importcpp: "af::wrap(@)", header: "arrayfire.h".}
proc sat*( af_in : AFArray ) : AFArray {.importcpp: "af::sat(@)", header: "arrayfire.h".}
proc ycbcr2rgb*( af_in : AFArray, standard : YCCStd ) : AFArray {.importcpp: "af::ycbcr2rgb(@)", header: "arrayfire.h".}
proc rgb2ycbcr*( af_in : AFArray, standard : YCCStd ) : AFArray {.importcpp: "af::rgb2ycbcr(@)", header: "arrayfire.h".}
proc moments*( af_out : float64, af_in : AFArray, moment : MomentType )  {.importcpp: "af::moments(@)", header: "arrayfire.h".}
proc moments*( af_in : AFArray, moment : MomentType ) : AFArray {.importcpp: "af::moments(@)", header: "arrayfire.h".}
proc canny*( af_in : AFArray, thresholdType : CannyThreshold, lowThresholdRatio : float32, highThresholdRatio : float32, sobelWindow : uint32, isFast : bool ) : AFArray {.importcpp: "af::canny(@)", header: "arrayfire.h".}
proc anisotropicDiffusion*( af_in : AFArray, timestep : float32, conductance : float32, iterations : uint32, fftype : FluxFunction, diffusionKind : DiffusionEq ) : AFArray {.importcpp: "af::anisotropicDiffusion(@)", header: "arrayfire.h".}
proc iterativeDeconv*( af_in : AFArray, ker : AFArray, iterations : uint32, relaxFactor : float32, algo : IterativeDeconvAlgo ) : AFArray {.importcpp: "af::iterativeDeconv(@)", header: "arrayfire.h".}
proc inverseDeconv*( af_in : AFArray, psf : AFArray, gamma : float32, algo : InverseDeconvAlgo ) : AFArray {.importcpp: "af::inverseDeconv(@)", header: "arrayfire.h".}
proc confidenceCC*( af_in : AFArray, seeds : AFArray, radius : uint32, multiplier : uint32, iter : int32, segmentedValue : float64 ) : AFArray {.importcpp: "af::confidenceCC(@)", header: "arrayfire.h".}
proc confidenceCC*( af_in : AFArray, seedx : AFArray, seedy : AFArray, radius : uint32, multiplier : uint32, iter : int32, segmentedValue : float64 ) : AFArray {.importcpp: "af::confidenceCC(@)", header: "arrayfire.h".}
proc confidenceCC*( af_in : AFArray, num_seeds : uint32, seedx : uint32, seedy : uint32, radius : uint32, multiplier : uint32, iter : int32, segmentedValue : float64 ) : AFArray {.importcpp: "af::confidenceCC(@)", header: "arrayfire.h".}
proc svd*( u : AFArray, s : AFArray, vt : AFArray, af_in : AFArray )  {.importcpp: "af::svd(@)", header: "arrayfire.h".}
proc svdInPlace*( u : AFArray, s : AFArray, vt : AFArray, af_in : AFArray )  {.importcpp: "af::svdInPlace(@)", header: "arrayfire.h".}
proc lu*( af_out : AFArray, pivot : AFArray, af_in : AFArray, is_lapack_piv : bool )  {.importcpp: "af::lu(@)", header: "arrayfire.h".}
proc lu*( lower : AFArray, upper : AFArray, pivot : AFArray, af_in : AFArray )  {.importcpp: "af::lu(@)", header: "arrayfire.h".}
proc luInPlace*( pivot : AFArray, af_in : AFArray, is_lapack_piv : bool )  {.importcpp: "af::luInPlace(@)", header: "arrayfire.h".}
proc qr*( af_out : AFArray, tau : AFArray, af_in : AFArray )  {.importcpp: "af::qr(@)", header: "arrayfire.h".}
proc qr*( q : AFArray, r : AFArray, tau : AFArray, af_in : AFArray )  {.importcpp: "af::qr(@)", header: "arrayfire.h".}
proc qrInPlace*( tau : AFArray, af_in : AFArray )  {.importcpp: "af::qrInPlace(@)", header: "arrayfire.h".}
proc cholesky*( af_out : AFArray, af_in : AFArray, is_upper : bool ) : int32 {.importcpp: "af::cholesky(@)", header: "arrayfire.h".}
proc choleskyInPlace*( af_in : AFArray, is_upper : bool ) : int32 {.importcpp: "af::choleskyInPlace(@)", header: "arrayfire.h".}
proc solve*( a : AFArray, b : AFArray, options : MatProp ) : AFArray {.importcpp: "af::solve(@)", header: "arrayfire.h".}
proc solveLU*( a : AFArray, piv : AFArray, b : AFArray, options : MatProp ) : AFArray {.importcpp: "af::solveLU(@)", header: "arrayfire.h".}
proc inverse*( af_in : AFArray, options : MatProp ) : AFArray {.importcpp: "af::inverse(@)", header: "arrayfire.h".}
proc pinverse*( af_in : AFArray, tol : float64, options : MatProp ) : AFArray {.importcpp: "af::pinverse(@)", header: "arrayfire.h".}
proc rank*( af_in : AFArray, tol : float64 ) : uint32 {.importcpp: "af::rank(@)", header: "arrayfire.h".}
proc norm*( af_in : AFArray, af_type : NormType, p : float64, q : float64 ) : float64 {.importcpp: "af::norm(@)", header: "arrayfire.h".}
proc isLAPACKAvailable*(  ) : bool {.importcpp: "af::isLAPACKAvailable(@)", header: "arrayfire.h".}
proc convolve2GradientNN*( incoming_gradient : AFArray, original_signal : AFArray, original_filter : AFArray, convolved_output : AFArray, stride : Dim4, padding : Dim4, dilation : Dim4, grad_type : ConvGradientType ) : AFArray {.importcpp: "af::convolve2GradientNN(@)", header: "arrayfire.h".}
proc randu*( dims : Dim4, ty : Dtype, r : RandomEngine ) : AFArray {.importcpp: "af::randu(@)", header: "arrayfire.h".}
proc randn*( dims : Dim4, ty : Dtype, r : RandomEngine ) : AFArray {.importcpp: "af::randn(@)", header: "arrayfire.h".}
proc randu*( dims : Dim4, ty : Dtype ) : AFArray {.importcpp: "af::randu(@)", header: "arrayfire.h".}
proc randu*( d0 : DimT, ty : Dtype ) : AFArray {.importcpp: "af::randu(@)", header: "arrayfire.h".}
proc randu*( d0 : DimT, d1 : DimT, ty : Dtype ) : AFArray {.importcpp: "af::randu(@)", header: "arrayfire.h".}
proc randu*( d0 : DimT, d1 : DimT, d2 : DimT, ty : Dtype ) : AFArray {.importcpp: "af::randu(@)", header: "arrayfire.h".}
proc randu*( d0 : DimT, d1 : DimT, d2 : DimT, d3 : DimT, ty : Dtype ) : AFArray {.importcpp: "af::randu(@)", header: "arrayfire.h".}
proc randn*( dims : Dim4, ty : Dtype ) : AFArray {.importcpp: "af::randn(@)", header: "arrayfire.h".}
proc randn*( d0 : DimT, ty : Dtype ) : AFArray {.importcpp: "af::randn(@)", header: "arrayfire.h".}
proc randn*( d0 : DimT, d1 : DimT, ty : Dtype ) : AFArray {.importcpp: "af::randn(@)", header: "arrayfire.h".}
proc randn*( d0 : DimT, d1 : DimT, d2 : DimT, ty : Dtype ) : AFArray {.importcpp: "af::randn(@)", header: "arrayfire.h".}
proc randn*( d0 : DimT, d1 : DimT, d2 : DimT, d3 : DimT, ty : Dtype ) : AFArray {.importcpp: "af::randn(@)", header: "arrayfire.h".}
proc setDefaultRandomEngineType*( rtype : RandomEngineType )  {.importcpp: "af::setDefaultRandomEngineType(@)", header: "arrayfire.h".}
proc getDefaultRandomEngine*(  ) : RandomEngine {.importcpp: "af::getDefaultRandomEngine(@)", header: "arrayfire.h".}
proc setSeed*( seed : uint64 )  {.importcpp: "af::setSeed(@)", header: "arrayfire.h".}
proc getSeed*(  ) : uint64 {.importcpp: "af::getSeed(@)", header: "arrayfire.h".}
proc approx1*( af_in : AFArray, pos : AFArray, af_mathod : InterpType, off_grid : float32 ) : AFArray {.importcpp: "af::approx1(@)", header: "arrayfire.h".}
proc approx2*( af_in : AFArray, pos0 : AFArray, pos1 : AFArray, af_mathod : InterpType, off_grid : float32 ) : AFArray {.importcpp: "af::approx2(@)", header: "arrayfire.h".}
proc approx1*( af_in : AFArray, pos : AFArray, interp_dim : int32, idx_start : float64, idx_step : float64, af_mathod : InterpType, off_grid : float32 ) : AFArray {.importcpp: "af::approx1(@)", header: "arrayfire.h".}
proc approx2*( af_in : AFArray, pos0 : AFArray, interp_dim0 : int32, idx_start_dim0 : float64, idx_step_dim0 : float64, pos1 : AFArray, interp_dim1 : int32, idx_start_dim1 : float64, idx_step_dim1 : float64, af_mathod : InterpType, off_grid : float32 ) : AFArray {.importcpp: "af::approx2(@)", header: "arrayfire.h".}
proc fftNorm*( af_in : AFArray, norm_factor : float64, odim0 : DimT ) : AFArray {.importcpp: "af::fftNorm(@)", header: "arrayfire.h".}
proc fft2Norm*( af_in : AFArray, norm_factor : float64, odim0 : DimT, odim1 : DimT ) : AFArray {.importcpp: "af::fft2Norm(@)", header: "arrayfire.h".}
proc fft3Norm*( af_in : AFArray, norm_factor : float64, odim0 : DimT, odim1 : DimT, odim2 : DimT ) : AFArray {.importcpp: "af::fft3Norm(@)", header: "arrayfire.h".}
proc fftInPlace*( af_in : AFArray, norm_factor : float64 )  {.importcpp: "af::fftInPlace(@)", header: "arrayfire.h".}
proc fft2InPlace*( af_in : AFArray, norm_factor : float64 )  {.importcpp: "af::fft2InPlace(@)", header: "arrayfire.h".}
proc fft3InPlace*( af_in : AFArray, norm_factor : float64 )  {.importcpp: "af::fft3InPlace(@)", header: "arrayfire.h".}
proc fft*( af_in : AFArray, odim0 : DimT ) : AFArray {.importcpp: "af::fft(@)", header: "arrayfire.h".}
proc fft2*( af_in : AFArray, odim0 : DimT, odim1 : DimT ) : AFArray {.importcpp: "af::fft2(@)", header: "arrayfire.h".}
proc fft3*( af_in : AFArray, odim0 : DimT, odim1 : DimT, odim2 : DimT ) : AFArray {.importcpp: "af::fft3(@)", header: "arrayfire.h".}
proc dft*( af_in : AFArray, norm_factor : float64, outDims : Dim4 ) : AFArray {.importcpp: "af::dft(@)", header: "arrayfire.h".}
proc dft*( af_in : AFArray, outDims : Dim4 ) : AFArray {.importcpp: "af::dft(@)", header: "arrayfire.h".}
proc dft*( af_in : AFArray ) : AFArray {.importcpp: "af::dft(@)", header: "arrayfire.h".}
proc ifftNorm*( af_in : AFArray, norm_factor : float64, odim0 : DimT ) : AFArray {.importcpp: "af::ifftNorm(@)", header: "arrayfire.h".}
proc ifft2Norm*( af_in : AFArray, norm_factor : float64, odim0 : DimT, odim1 : DimT ) : AFArray {.importcpp: "af::ifft2Norm(@)", header: "arrayfire.h".}
proc ifft3Norm*( af_in : AFArray, norm_factor : float64, odim0 : DimT, odim1 : DimT, odim2 : DimT ) : AFArray {.importcpp: "af::ifft3Norm(@)", header: "arrayfire.h".}
proc ifftInPlace*( af_in : AFArray, norm_factor : float64 )  {.importcpp: "af::ifftInPlace(@)", header: "arrayfire.h".}
proc ifft2InPlace*( af_in : AFArray, norm_factor : float64 )  {.importcpp: "af::ifft2InPlace(@)", header: "arrayfire.h".}
proc ifft3InPlace*( af_in : AFArray, norm_factor : float64 )  {.importcpp: "af::ifft3InPlace(@)", header: "arrayfire.h".}
proc ifft*( af_in : AFArray, odim0 : DimT ) : AFArray {.importcpp: "af::ifft(@)", header: "arrayfire.h".}
proc ifft2*( af_in : AFArray, odim0 : DimT, odim1 : DimT ) : AFArray {.importcpp: "af::ifft2(@)", header: "arrayfire.h".}
proc ifft3*( af_in : AFArray, odim0 : DimT, odim1 : DimT, odim2 : DimT ) : AFArray {.importcpp: "af::ifft3(@)", header: "arrayfire.h".}
proc idft*( af_in : AFArray, norm_factor : float64, outDims : Dim4 ) : AFArray {.importcpp: "af::idft(@)", header: "arrayfire.h".}
proc idft*( af_in : AFArray, outDims : Dim4 ) : AFArray {.importcpp: "af::idft(@)", header: "arrayfire.h".}
proc idft*( af_in : AFArray ) : AFArray {.importcpp: "af::idft(@)", header: "arrayfire.h".}
proc convolve*( signal : AFArray, filter : AFArray, mode : ConvMode, domain : ConvDomain ) : AFArray {.importcpp: "af::convolve(@)", header: "arrayfire.h".}
proc convolve*( col_filter : AFArray, row_filter : AFArray, signal : AFArray, mode : ConvMode ) : AFArray {.importcpp: "af::convolve(@)", header: "arrayfire.h".}
proc convolve1*( signal : AFArray, filter : AFArray, mode : ConvMode, domain : ConvDomain ) : AFArray {.importcpp: "af::convolve1(@)", header: "arrayfire.h".}
proc convolve2*( signal : AFArray, filter : AFArray, mode : ConvMode, domain : ConvDomain ) : AFArray {.importcpp: "af::convolve2(@)", header: "arrayfire.h".}
proc convolve2NN*( signal : AFArray, filter : AFArray, stride : Dim4, padding : Dim4, dilation : Dim4 ) : AFArray {.importcpp: "af::convolve2NN(@)", header: "arrayfire.h".}
proc convolve3*( signal : AFArray, filter : AFArray, mode : ConvMode, domain : ConvDomain ) : AFArray {.importcpp: "af::convolve3(@)", header: "arrayfire.h".}
proc fftConvolve*( signal : AFArray, filter : AFArray, mode : ConvMode ) : AFArray {.importcpp: "af::fftConvolve(@)", header: "arrayfire.h".}
proc fftConvolve1*( signal : AFArray, filter : AFArray, mode : ConvMode ) : AFArray {.importcpp: "af::fftConvolve1(@)", header: "arrayfire.h".}
proc fftConvolve2*( signal : AFArray, filter : AFArray, mode : ConvMode ) : AFArray {.importcpp: "af::fftConvolve2(@)", header: "arrayfire.h".}
proc fftConvolve3*( signal : AFArray, filter : AFArray, mode : ConvMode ) : AFArray {.importcpp: "af::fftConvolve3(@)", header: "arrayfire.h".}
proc fir*( b : AFArray, x : AFArray ) : AFArray {.importcpp: "af::fir(@)", header: "arrayfire.h".}
proc iir*( b : AFArray, a : AFArray, x : AFArray ) : AFArray {.importcpp: "af::iir(@)", header: "arrayfire.h".}
proc medfilt*( af_in : AFArray, wind_length : DimT, wind_width : DimT, edge_pad : BorderType ) : AFArray {.importcpp: "af::medfilt(@)", header: "arrayfire.h".}
proc medfilt1*( af_in : AFArray, wind_width : DimT, edge_pad : BorderType ) : AFArray {.importcpp: "af::medfilt1(@)", header: "arrayfire.h".}
proc medfilt2*( af_in : AFArray, wind_length : DimT, wind_width : DimT, edge_pad : BorderType ) : AFArray {.importcpp: "af::medfilt2(@)", header: "arrayfire.h".}
proc setFFTPlanCacheSize*( cacheSize : uint32 )  {.importcpp: "af::setFFTPlanCacheSize(@)", header: "arrayfire.h".}
proc sparse*( nRows : DimT, nCols : DimT, values : AFArray, rowIdx : AFArray, colIdx : AFArray, stype : Storage ) : AFArray {.importcpp: "af::sparse(@)", header: "arrayfire.h".}
proc sparse*( nRows : DimT, nCols : DimT, nNZ : DimT, values : pointer, rowIdx : int32, colIdx : int32, af_type : Dtype, stype : Storage, src : Source ) : AFArray {.importcpp: "af::sparse(@)", header: "arrayfire.h".}
proc sparse*( dense : AFArray, stype : Storage ) : AFArray {.importcpp: "af::sparse(@)", header: "arrayfire.h".}
proc sparseConvertTo*( af_in : AFArray, destStrorage : Storage ) : AFArray {.importcpp: "af::sparseConvertTo(@)", header: "arrayfire.h".}
proc dense*( sparse : AFArray ) : AFArray {.importcpp: "af::dense(@)", header: "arrayfire.h".}
proc sparseGetInfo*( values : AFArray, rowIdx : AFArray, colIdx : AFArray, stype : Storage, af_in : AFArray )  {.importcpp: "af::sparseGetInfo(@)", header: "arrayfire.h".}
proc sparseGetValues*( af_in : AFArray ) : AFArray {.importcpp: "af::sparseGetValues(@)", header: "arrayfire.h".}
proc sparseGetRowIdx*( af_in : AFArray ) : AFArray {.importcpp: "af::sparseGetRowIdx(@)", header: "arrayfire.h".}
proc sparseGetColIdx*( af_in : AFArray ) : AFArray {.importcpp: "af::sparseGetColIdx(@)", header: "arrayfire.h".}
proc sparseGetNNZ*( af_in : AFArray ) : DimT {.importcpp: "af::sparseGetNNZ(@)", header: "arrayfire.h".}
proc sparseGetStorage*( af_in : AFArray ) : Storage {.importcpp: "af::sparseGetStorage(@)", header: "arrayfire.h".}
proc mean*( af_in : AFArray, dim : DimT ) : AFArray {.importcpp: "af::mean(@)", header: "arrayfire.h".}
proc mean*( af_in : AFArray, weights : AFArray, dim : DimT ) : AFArray {.importcpp: "af::mean(@)", header: "arrayfire.h".}
proc af_var*( af_in : AFArray, isbiased : bool, dim : DimT ) : AFArray {.importcpp: "af::var(@)", header: "arrayfire.h".}
proc af_var*( af_in : AFArray, bias : VarBias, dim : DimT ) : AFArray {.importcpp: "af::var(@)", header: "arrayfire.h".}
proc af_var*( af_in : AFArray, weights : AFArray, dim : DimT ) : AFArray {.importcpp: "af::var(@)", header: "arrayfire.h".}
proc meanvar*( mean : AFArray, af_var : AFArray, af_in : AFArray, weights : AFArray, bias : VarBias, dim : DimT )  {.importcpp: "af::meanvar(@)", header: "arrayfire.h".}
proc stdev*( af_in : AFArray, dim : DimT ) : AFArray {.importcpp: "af::stdev(@)", header: "arrayfire.h".}
proc stdev*( af_in : AFArray, bias : VarBias, dim : DimT ) : AFArray {.importcpp: "af::stdev(@)", header: "arrayfire.h".}
proc cov*( X : AFArray, Y : AFArray, isbiased : bool ) : AFArray {.importcpp: "af::cov(@)", header: "arrayfire.h".}
proc cov*( X : AFArray, Y : AFArray, bias : VarBias ) : AFArray {.importcpp: "af::cov(@)", header: "arrayfire.h".}
proc median*( af_in : AFArray, dim : DimT ) : AFArray {.importcpp: "af::median(@)", header: "arrayfire.h".}
proc topk*( values : AFArray, indices : AFArray, af_in : AFArray, k : int32, dim : int32, order : TopkFunction )  {.importcpp: "af::topk(@)", header: "arrayfire.h".}
proc fast*( af_in : AFArray, thr : float32, arc_length : uint32, non_max : bool, feature_ratio : float32, edge : uint32 ) : Features {.importcpp: "af::fast(@)", header: "arrayfire.h".}
proc harris*( af_in : AFArray, max_corners : uint32, min_response : float32, sigma : float32, block_size : uint32, k_thr : float32 ) : Features {.importcpp: "af::harris(@)", header: "arrayfire.h".}
proc orb*( feat : Features, desc : AFArray, image : AFArray, fast_thr : float32, max_feat : uint32, scl_fctr : float32, levels : uint32, blur_img : bool )  {.importcpp: "af::orb(@)", header: "arrayfire.h".}
proc sift*( feat : Features, desc : AFArray, af_in : AFArray, n_layers : uint32, contrast_thr : float32, edge_thr : float32, init_sigma : float32, double_input : bool, intensity_scale : float32, feature_ratio : float32 )  {.importcpp: "af::sift(@)", header: "arrayfire.h".}
proc gloh*( feat : Features, desc : AFArray, af_in : AFArray, n_layers : uint32, contrast_thr : float32, edge_thr : float32, init_sigma : float32, double_input : bool, intensity_scale : float32, feature_ratio : float32 )  {.importcpp: "af::gloh(@)", header: "arrayfire.h".}
proc hammingMatcher*( idx : AFArray, dist : AFArray, query : AFArray, train : AFArray, dist_dim : DimT, n_dist : uint32 )  {.importcpp: "af::hammingMatcher(@)", header: "arrayfire.h".}
proc nearestNeighbour*( idx : AFArray, dist : AFArray, query : AFArray, train : AFArray, dist_dim : DimT, n_dist : uint32, dist_type : MatchType )  {.importcpp: "af::nearestNeighbour(@)", header: "arrayfire.h".}
proc matchTemplate*( searchImg : AFArray, templateImg : AFArray, mType : MatchType ) : AFArray {.importcpp: "af::matchTemplate(@)", header: "arrayfire.h".}
proc susan*( af_in : AFArray, radius : uint32, diff_thr : float32, geom_thr : float32, feature_ratio : float32, edge : uint32 ) : Features {.importcpp: "af::susan(@)", header: "arrayfire.h".}
proc dog*( af_in : AFArray, radius1 : int32, radius2 : int32 ) : AFArray {.importcpp: "af::dog(@)", header: "arrayfire.h".}
proc homography*( H : AFArray, inliers : int32, x_src : AFArray, y_src : AFArray, x_dst : AFArray, y_dst : AFArray, htype : HomographyType, inlier_thr : float32, iterations : uint32, otype : Dtype )  {.importcpp: "af::homography(@)", header: "arrayfire.h".}
#endregion

 
#region Methods
proc af_seq*( length : float64 ) : AF_Seq {.constructor, importcpp: "af::seq(@)", header: "arrayfire.h".}
proc destroy_seq*(this: var AF_Seq) {.importcpp: "#.~seq()", header : "arrayfire.h".}
proc af_seq*( begin : float64, af_end : float64, step : float64 ) : AF_Seq {.constructor, importcpp: "af::seq(@)", header: "arrayfire.h".}
proc af_seq*( other : AF_Seq, is_gfor : bool ) : AF_Seq {.constructor, importcpp: "af::seq(@)", header: "arrayfire.h".}
proc af_seq*( s : AF_Seq ) : AF_Seq {.constructor, importcpp: "af::seq(@)", header: "arrayfire.h".}
proc `assign`*(this : AF_Seq, s : AF_Seq)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc `-`*(this : AF_Seq) : AF_Seq {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `+`*(this : AF_Seq, x : float64) : AF_Seq {.importcpp: "(# + #)", header: "arrayfire.h".}
proc `-`*(this : AF_Seq, x : float64) : AF_Seq {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `*`*(this : AF_Seq, x : float64) : AF_Seq {.importcpp: "(# * #)", header: "arrayfire.h".}
proc `afa`*(this : AF_Seq) : AFArray {.importcpp: "(# afa #)", header: "arrayfire.h".}
proc init*( this : AF_Seq, begin : float64, af_end : float64, step : float64 )  {.importcpp: "init", header : "arrayfire.h".}
proc af_seq*( p : AF_Seq ) : AF_Seq {.constructor, importcpp: "af::seq(@)", header: "arrayfire.h".}
proc dim4*(  ) : Dim4 {.constructor, importcpp: "af::dim4(@)", header: "arrayfire.h".}
proc dim4*( first : DimT, second : DimT, third : DimT, fourth : DimT ) : Dim4 {.constructor, importcpp: "af::dim4(@)", header: "arrayfire.h".}
proc dim4*( other : Dim4 ) : Dim4 {.constructor, importcpp: "af::dim4(@)", header: "arrayfire.h".}
proc `assign`*(this : Dim4, other : Dim4)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc dim4*( ndims : uint32, dims : DimT ) : Dim4 {.constructor, importcpp: "af::dim4(@)", header: "arrayfire.h".}
proc elements*( this : Dim4 ) : DimT {.importcpp: "elements", header : "arrayfire.h".}
proc ndims*( this : Dim4 ) : DimT {.importcpp: "ndims", header : "arrayfire.h".}
proc `==`*(this : Dim4, other : Dim4) : bool {.importcpp: "(# == #)", header: "arrayfire.h".}
proc `!=`*(this : Dim4, other : Dim4)  {.importcpp: "(# != #)", header: "arrayfire.h".}
proc `*=`*(this : Dim4, other : Dim4)  {.importcpp: "(# *= #)", header: "arrayfire.h".}
proc `+=`*(this : Dim4, other : Dim4)  {.importcpp: "(# += #)", header: "arrayfire.h".}
proc `-=`*(this : Dim4, other : Dim4)  {.importcpp: "(# -= #)", header: "arrayfire.h".}
proc get*( this : Dim4 ) : DimT {.importcpp: "get", header : "arrayfire.h".}
proc err*( this : AF_Exception ) : Err {.importcpp: "af::err", header : "arrayfire.h".}
proc exception*(  ) : AF_Exception {.constructor, importcpp: "af::exception(@)", header: "arrayfire.h".}
proc exception*( msg : cstring ) : AF_Exception {.constructor, importcpp: "af::exception(@)", header: "arrayfire.h".}
proc exception*( file : cstring, line : uint32, err : Err ) : AF_Exception {.constructor, importcpp: "af::exception(@)", header: "arrayfire.h".}
proc exception*( msg : cstring, file : cstring, line : uint32, err : Err ) : AF_Exception {.constructor, importcpp: "af::exception(@)", header: "arrayfire.h".}
proc exception*( msg : cstring, af_func : cstring, file : cstring, line : uint32, err : Err ) : AF_Exception {.constructor, importcpp: "af::exception(@)", header: "arrayfire.h".}
proc destroy_exception*(this: var AF_Exception) {.importcpp: "#.~exception()", header : "arrayfire.h".}
proc what*( this : AF_Exception ) : cstring {.importcpp: "what", header : "arrayfire.h".}
proc exception*( p : AF_Exception ) : AF_Exception {.constructor, importcpp: "af::exception(@)", header: "arrayfire.h".}
proc `assign`*(this : AF_Exception, p : AF_Exception)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc index*(  ) : IndexT {.constructor, importcpp: "af::index(@)", header: "arrayfire.h".}
proc destroy_index*(this: var IndexT) {.importcpp: "#.~index()", header : "arrayfire.h".}
proc index*( idx : int32 ) : IndexT {.constructor, importcpp: "af::index(@)", header: "arrayfire.h".}
proc index*( s0 : AF_Seq ) : IndexT {.constructor, importcpp: "af::index(@)", header: "arrayfire.h".}
proc index*( idx0 : AFArray ) : IndexT {.constructor, importcpp: "af::index(@)", header: "arrayfire.h".}
proc index*( idx0 : IndexT ) : IndexT {.constructor, importcpp: "af::index(@)", header: "arrayfire.h".}
proc isspan*( this : IndexT ) : bool {.importcpp: "isspan", header : "arrayfire.h".}
proc get*( this : IndexT ) : IndexT {.importcpp: "get", header : "arrayfire.h".}
proc `assign`*(this : IndexT, idx0 : IndexT)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc set*( this : AFArray, tmp : AF_Array_Handle )  {.importcpp: "set", header : "arrayfire.h".}
proc af_array*(  ) : AFArray {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}
proc af_array*( other : AFArray ) : AFArray {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}
proc `assign`*(this : AFArray, other : AFArray)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc af_array*( handle : AF_Array_Handle ) : AFArray {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}
proc af_array*( af_in : AFArray ) : AFArray {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}
proc af_array*( dim0 : DimT, ty : Dtype ) : AFArray {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}
proc af_array*( dim0 : DimT, dim1 : DimT, ty : Dtype ) : AFArray {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}
proc af_array*( dim0 : DimT, dim1 : DimT, dim2 : DimT, ty : Dtype ) : AFArray {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}
proc af_array*( dim0 : DimT, dim1 : DimT, dim2 : DimT, dim3 : DimT, ty : Dtype ) : AFArray {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}
proc af_array*( dims : Dim4, ty : Dtype ) : AFArray {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}
proc af_array*( input : AFArray, dims : Dim4 ) : AFArray {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}
proc af_array*( input : AFArray, dim0 : DimT, dim1 : DimT, dim2 : DimT, dim3 : DimT ) : AFArray {.constructor, importcpp: "af::array(@)", header: "arrayfire.h".}
proc get*( this : AFArray ) : AF_Array_Handle {.importcpp: "get", header : "arrayfire.h".}
proc elements*( this : AFArray ) : DimT {.importcpp: "elements", header : "arrayfire.h".}
proc host*( this : AFArray, af_ptr : pointer )  {.importcpp: "host", header : "arrayfire.h".}
proc dtype*( this : AFArray ) : Dtype {.importcpp: "type", header : "arrayfire.h".}
proc dims*( this : AFArray ) : Dim4 {.importcpp: "dims", header : "arrayfire.h".}
proc dims*( this : AFArray, dim : uint32 ) : DimT {.importcpp: "dims", header : "arrayfire.h".}
proc numdims*( this : AFArray ) : uint32 {.importcpp: "numdims", header : "arrayfire.h".}
proc bytes*( this : AFArray ) : uint32 {.importcpp: "bytes", header : "arrayfire.h".}
proc allocated*( this : AFArray ) : uint32 {.importcpp: "allocated", header : "arrayfire.h".}
proc copy*( this : AFArray ) : AFArray {.importcpp: "copy", header : "arrayfire.h".}
proc isempty*( this : AFArray ) : bool {.importcpp: "isempty", header : "arrayfire.h".}
proc isscalar*( this : AFArray ) : bool {.importcpp: "isscalar", header : "arrayfire.h".}
proc isvector*( this : AFArray ) : bool {.importcpp: "isvector", header : "arrayfire.h".}
proc isrow*( this : AFArray ) : bool {.importcpp: "isrow", header : "arrayfire.h".}
proc iscolumn*( this : AFArray ) : bool {.importcpp: "iscolumn", header : "arrayfire.h".}
proc iscomplex*( this : AFArray ) : bool {.importcpp: "iscomplex", header : "arrayfire.h".}
proc isreal*( this : AFArray ) : bool {.importcpp: "isreal", header : "arrayfire.h".}
proc isdouble*( this : AFArray ) : bool {.importcpp: "isdouble", header : "arrayfire.h".}
proc issingle*( this : AFArray ) : bool {.importcpp: "issingle", header : "arrayfire.h".}
proc ishalf*( this : AFArray ) : bool {.importcpp: "ishalf", header : "arrayfire.h".}
proc isrealfloating*( this : AFArray ) : bool {.importcpp: "isrealfloating", header : "arrayfire.h".}
proc isfloating*( this : AFArray ) : bool {.importcpp: "isfloating", header : "arrayfire.h".}
proc isinteger*( this : AFArray ) : bool {.importcpp: "isinteger", header : "arrayfire.h".}
proc isbool*( this : AFArray ) : bool {.importcpp: "isbool", header : "arrayfire.h".}
proc issparse*( this : AFArray ) : bool {.importcpp: "issparse", header : "arrayfire.h".}
proc eval*( this : AFArray )  {.importcpp: "eval", header : "arrayfire.h".}
proc `call`*(this : AFArray, s0 : IndexT) : AFArray {.importcpp: "(# () #)", header: "arrayfire.h".}
proc `call`*(this : AFArray, s0 : IndexT, s1 : IndexT, s2 : IndexT, s3 : IndexT) : AFArray {.importcpp: "(# () #)", header: "arrayfire.h".}
proc row*( this : AFArray, index : int32 ) : AFArray {.importcpp: "row", header : "arrayfire.h".}
proc rows*( this : AFArray, first : int32, last : int32 ) : AFArray {.importcpp: "rows", header : "arrayfire.h".}
proc col*( this : AFArray, index : int32 ) : AFArray {.importcpp: "col", header : "arrayfire.h".}
proc cols*( this : AFArray, first : int32, last : int32 ) : AFArray {.importcpp: "cols", header : "arrayfire.h".}
proc slice*( this : AFArray, index : int32 ) : AFArray {.importcpp: "slice", header : "arrayfire.h".}
proc slices*( this : AFArray, first : int32, last : int32 ) : AFArray {.importcpp: "slices", header : "arrayfire.h".}
proc af_as*( this : AFArray, af_type : Dtype ) : AFArray {.importcpp: "as", header : "arrayfire.h".}
proc destroy_array*(this: var AFArray) {.importcpp: "#.~array()", header : "arrayfire.h".}
proc T*( this : AFArray ) : AFArray {.importcpp: "T", header : "arrayfire.h".}
proc H*( this : AFArray ) : AFArray {.importcpp: "H", header : "arrayfire.h".}
proc `assign`*(this : AFArray, val : AFArray)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc `assign`*(this : AFArray, val : float64)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc `assign`*(this : AFArray, val : float32)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc `assign`*(this : AFArray, val : int32)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc `assign`*(this : AFArray, val : uint32)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc `assign`*(this : AFArray, val : bool)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc `assign`*(this : AFArray, val : cstring)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc `assign`*(this : AFArray, val : int64)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc `assign`*(this : AFArray, val : uint64)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc `assign`*(this : AFArray, val : int16)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc `assign`*(this : AFArray, val : uint16)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc `+=`*(this : AFArray, val : AFArray)  {.importcpp: "(# += #)", header: "arrayfire.h".}
proc `+=`*(this : AFArray, val : float64)  {.importcpp: "(# += #)", header: "arrayfire.h".}
proc `+=`*(this : AFArray, val : float32)  {.importcpp: "(# += #)", header: "arrayfire.h".}
proc `+=`*(this : AFArray, val : int32)  {.importcpp: "(# += #)", header: "arrayfire.h".}
proc `+=`*(this : AFArray, val : uint32)  {.importcpp: "(# += #)", header: "arrayfire.h".}
proc `+=`*(this : AFArray, val : bool)  {.importcpp: "(# += #)", header: "arrayfire.h".}
proc `+=`*(this : AFArray, val : cstring)  {.importcpp: "(# += #)", header: "arrayfire.h".}
proc `+=`*(this : AFArray, val : int64)  {.importcpp: "(# += #)", header: "arrayfire.h".}
proc `+=`*(this : AFArray, val : uint64)  {.importcpp: "(# += #)", header: "arrayfire.h".}
proc `+=`*(this : AFArray, val : int16)  {.importcpp: "(# += #)", header: "arrayfire.h".}
proc `+=`*(this : AFArray, val : uint16)  {.importcpp: "(# += #)", header: "arrayfire.h".}
proc `-=`*(this : AFArray, val : AFArray)  {.importcpp: "(# -= #)", header: "arrayfire.h".}
proc `-=`*(this : AFArray, val : float64)  {.importcpp: "(# -= #)", header: "arrayfire.h".}
proc `-=`*(this : AFArray, val : float32)  {.importcpp: "(# -= #)", header: "arrayfire.h".}
proc `-=`*(this : AFArray, val : int32)  {.importcpp: "(# -= #)", header: "arrayfire.h".}
proc `-=`*(this : AFArray, val : uint32)  {.importcpp: "(# -= #)", header: "arrayfire.h".}
proc `-=`*(this : AFArray, val : bool)  {.importcpp: "(# -= #)", header: "arrayfire.h".}
proc `-=`*(this : AFArray, val : cstring)  {.importcpp: "(# -= #)", header: "arrayfire.h".}
proc `-=`*(this : AFArray, val : int64)  {.importcpp: "(# -= #)", header: "arrayfire.h".}
proc `-=`*(this : AFArray, val : uint64)  {.importcpp: "(# -= #)", header: "arrayfire.h".}
proc `-=`*(this : AFArray, val : int16)  {.importcpp: "(# -= #)", header: "arrayfire.h".}
proc `-=`*(this : AFArray, val : uint16)  {.importcpp: "(# -= #)", header: "arrayfire.h".}
proc `*=`*(this : AFArray, val : AFArray)  {.importcpp: "(# *= #)", header: "arrayfire.h".}
proc `*=`*(this : AFArray, val : float64)  {.importcpp: "(# *= #)", header: "arrayfire.h".}
proc `*=`*(this : AFArray, val : float32)  {.importcpp: "(# *= #)", header: "arrayfire.h".}
proc `*=`*(this : AFArray, val : int32)  {.importcpp: "(# *= #)", header: "arrayfire.h".}
proc `*=`*(this : AFArray, val : uint32)  {.importcpp: "(# *= #)", header: "arrayfire.h".}
proc `*=`*(this : AFArray, val : bool)  {.importcpp: "(# *= #)", header: "arrayfire.h".}
proc `*=`*(this : AFArray, val : cstring)  {.importcpp: "(# *= #)", header: "arrayfire.h".}
proc `*=`*(this : AFArray, val : int64)  {.importcpp: "(# *= #)", header: "arrayfire.h".}
proc `*=`*(this : AFArray, val : uint64)  {.importcpp: "(# *= #)", header: "arrayfire.h".}
proc `*=`*(this : AFArray, val : int16)  {.importcpp: "(# *= #)", header: "arrayfire.h".}
proc `*=`*(this : AFArray, val : uint16)  {.importcpp: "(# *= #)", header: "arrayfire.h".}
proc `/=`*(this : AFArray, val : AFArray)  {.importcpp: "(# /= #)", header: "arrayfire.h".}
proc `/=`*(this : AFArray, val : float64)  {.importcpp: "(# /= #)", header: "arrayfire.h".}
proc `/=`*(this : AFArray, val : float32)  {.importcpp: "(# /= #)", header: "arrayfire.h".}
proc `/=`*(this : AFArray, val : int32)  {.importcpp: "(# /= #)", header: "arrayfire.h".}
proc `/=`*(this : AFArray, val : uint32)  {.importcpp: "(# /= #)", header: "arrayfire.h".}
proc `/=`*(this : AFArray, val : bool)  {.importcpp: "(# /= #)", header: "arrayfire.h".}
proc `/=`*(this : AFArray, val : cstring)  {.importcpp: "(# /= #)", header: "arrayfire.h".}
proc `/=`*(this : AFArray, val : int64)  {.importcpp: "(# /= #)", header: "arrayfire.h".}
proc `/=`*(this : AFArray, val : uint64)  {.importcpp: "(# /= #)", header: "arrayfire.h".}
proc `/=`*(this : AFArray, val : int16)  {.importcpp: "(# /= #)", header: "arrayfire.h".}
proc `/=`*(this : AFArray, val : uint16)  {.importcpp: "(# /= #)", header: "arrayfire.h".}
proc `-`*(this : AFArray) : AFArray {.importcpp: "(# - #)", header: "arrayfire.h".}
proc `!`*(this : AFArray) : AFArray {.importcpp: "(# ! #)", header: "arrayfire.h".}
proc `~`*(this : AFArray) : AFArray {.importcpp: "(# ~ #)", header: "arrayfire.h".}
proc nonzeros*( this : AFArray ) : int32 {.importcpp: "nonzeros", header : "arrayfire.h".}
proc lock*( this : AFArray )  {.importcpp: "lock", header : "arrayfire.h".}
proc isLocked*( this : AFArray ) : bool {.importcpp: "isLocked", header : "arrayfire.h".}
proc unlock*( this : AFArray )  {.importcpp: "unlock", header : "arrayfire.h".}
proc event*( e : Event ) : Event {.constructor, importcpp: "af::event(@)", header: "arrayfire.h".}
proc event*( other : Event ) : Event {.constructor, importcpp: "af::event(@)", header: "arrayfire.h".}
proc `assign`*(this : Event, other : Event)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc event*(  ) : Event {.constructor, importcpp: "af::event(@)", header: "arrayfire.h".}
proc destroy_event*(this: var Event) {.importcpp: "#.~event()", header : "arrayfire.h".}
proc get*( this : Event ) : Event {.importcpp: "get", header : "arrayfire.h".}
proc mark*( this : Event )  {.importcpp: "mark", header : "arrayfire.h".}
proc enqueue*( this : Event )  {.importcpp: "enqueue", header : "arrayfire.h".}
proc af_block*( this : Event )  {.importcpp: "block", header : "arrayfire.h".}
proc features*(  ) : Features {.constructor, importcpp: "af::features(@)", header: "arrayfire.h".}
proc features*( n : uint32 ) : Features {.constructor, importcpp: "af::features(@)", header: "arrayfire.h".}
proc features*( f : Features ) : Features {.constructor, importcpp: "af::features(@)", header: "arrayfire.h".}
proc destroy_features*(this: var Features) {.importcpp: "#.~features()", header : "arrayfire.h".}
proc `assign`*(this : Features, other : Features)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc features*( other : Features ) : Features {.constructor, importcpp: "af::features(@)", header: "arrayfire.h".}
proc getNumFeatures*( this : Features ) : uint32 {.importcpp: "getNumFeatures", header : "arrayfire.h".}
proc getX*( this : Features ) : AFArray {.importcpp: "getX", header : "arrayfire.h".}
proc getY*( this : Features ) : AFArray {.importcpp: "getY", header : "arrayfire.h".}
proc getScore*( this : Features ) : AFArray {.importcpp: "getScore", header : "arrayfire.h".}
proc getOrientation*( this : Features ) : AFArray {.importcpp: "getOrientation", header : "arrayfire.h".}
proc getSize*( this : Features ) : AFArray {.importcpp: "getSize", header : "arrayfire.h".}
proc get*( this : Features ) : Features {.importcpp: "get", header : "arrayfire.h".}
proc initWindow*( this : Window, width : int32, height : int32, title : cstring )  {.importcpp: "af::initWindow", header : "arrayfire.h".}
proc make_window*( p : Window ) : Window {.constructor, importcpp: "af::Window(@)", header: "arrayfire.h".}
proc `assign`*(this : Window, p : Window)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc make_window*(  ) : Window {.constructor, importcpp: "af::Window(@)", header: "arrayfire.h".}
proc make_window*( title : cstring ) : Window {.constructor, importcpp: "af::Window(@)", header: "arrayfire.h".}
proc make_window*( width : int32, height : int32, title : cstring ) : Window {.constructor, importcpp: "af::Window(@)", header: "arrayfire.h".}
proc make_window*( window : Window ) : Window {.constructor, importcpp: "af::Window(@)", header: "arrayfire.h".}
proc destroy_Window*(this: var Window) {.importcpp: "#.~Window()", header : "arrayfire.h".}
proc get*( this : Window ) : Window {.importcpp: "get", header : "arrayfire.h".}
proc setPos*( this : Window, x : uint32, y : uint32 )  {.importcpp: "setPos", header : "arrayfire.h".}
proc setTitle*( this : Window, title : cstring )  {.importcpp: "setTitle", header : "arrayfire.h".}
proc setSize*( this : Window, w : uint32, h : uint32 )  {.importcpp: "setSize", header : "arrayfire.h".}
proc setColorMap*( this : Window, cmap : ColorMap )  {.importcpp: "setColorMap", header : "arrayfire.h".}
proc image*( this : Window, af_in : AFArray, title : cstring )  {.importcpp: "image", header : "arrayfire.h".}
proc plot3*( this : Window, af_in : AFArray, title : cstring )  {.importcpp: "plot3", header : "arrayfire.h".}
proc plot*( this : Window, af_in : AFArray, title : cstring )  {.importcpp: "plot", header : "arrayfire.h".}
proc plot*( this : Window, X : AFArray, Y : AFArray, Z : AFArray, title : cstring )  {.importcpp: "plot", header : "arrayfire.h".}
proc plot*( this : Window, X : AFArray, Y : AFArray, title : cstring )  {.importcpp: "plot", header : "arrayfire.h".}
proc scatter*( this : Window, af_in : AFArray, marker : MarkerType, title : cstring )  {.importcpp: "scatter", header : "arrayfire.h".}
proc scatter*( this : Window, X : AFArray, Y : AFArray, Z : AFArray, marker : MarkerType, title : cstring )  {.importcpp: "scatter", header : "arrayfire.h".}
proc scatter*( this : Window, X : AFArray, Y : AFArray, marker : MarkerType, title : cstring )  {.importcpp: "scatter", header : "arrayfire.h".}
proc scatter3*( this : Window, P : AFArray, marker : MarkerType, title : cstring )  {.importcpp: "scatter3", header : "arrayfire.h".}
proc hist*( this : Window, X : AFArray, minval : float64, maxval : float64, title : cstring )  {.importcpp: "hist", header : "arrayfire.h".}
proc surface*( this : Window, S : AFArray, title : cstring )  {.importcpp: "surface", header : "arrayfire.h".}
proc surface*( this : Window, xVals : AFArray, yVals : AFArray, S : AFArray, title : cstring )  {.importcpp: "surface", header : "arrayfire.h".}
proc vectorField*( this : Window, points : AFArray, directions : AFArray, title : cstring )  {.importcpp: "vectorField", header : "arrayfire.h".}
proc vectorField*( this : Window, xPoints : AFArray, yPoints : AFArray, zPoints : AFArray, xDirs : AFArray, yDirs : AFArray, zDirs : AFArray, title : cstring )  {.importcpp: "vectorField", header : "arrayfire.h".}
proc vectorField*( this : Window, xPoints : AFArray, yPoints : AFArray, xDirs : AFArray, yDirs : AFArray, title : cstring )  {.importcpp: "vectorField", header : "arrayfire.h".}
proc setAxesLimits*( this : Window, x : AFArray, y : AFArray, exact : bool )  {.importcpp: "setAxesLimits", header : "arrayfire.h".}
proc setAxesLimits*( this : Window, x : AFArray, y : AFArray, z : AFArray, exact : bool )  {.importcpp: "setAxesLimits", header : "arrayfire.h".}
proc setAxesLimits*( this : Window, xmin : float32, xmax : float32, ymin : float32, ymax : float32, exact : bool )  {.importcpp: "setAxesLimits", header : "arrayfire.h".}
proc setAxesLimits*( this : Window, xmin : float32, xmax : float32, ymin : float32, ymax : float32, zmin : float32, zmax : float32, exact : bool )  {.importcpp: "setAxesLimits", header : "arrayfire.h".}
proc setAxesTitles*( this : Window, xtitle : cstring, ytitle : cstring, ztitle : cstring )  {.importcpp: "setAxesTitles", header : "arrayfire.h".}
proc setAxesLabelFormat*( this : Window, xformat : cstring, yformat : cstring, zformat : cstring )  {.importcpp: "setAxesLabelFormat", header : "arrayfire.h".}
proc grid*( this : Window, rows : int32, cols : int32 )  {.importcpp: "grid", header : "arrayfire.h".}
proc show*( this : Window )  {.importcpp: "show", header : "arrayfire.h".}
proc close*( this : Window ) : bool {.importcpp: "close", header : "arrayfire.h".}
proc setVisibility*( this : Window, isVisible : bool )  {.importcpp: "setVisibility", header : "arrayfire.h".}
proc `call`*(this : Window, r : int32, c : int32) : Window {.importcpp: "(# () #)", header: "arrayfire.h".}
proc randomEngine*( typeIn : RandomEngineType, seedIn : uint64 ) : RandomEngine {.constructor, importcpp: "af::randomEngine(@)", header: "arrayfire.h".}
proc randomEngine*( other : RandomEngine ) : RandomEngine {.constructor, importcpp: "af::randomEngine(@)", header: "arrayfire.h".}
proc randomEngine*( engine : RandomEngine ) : RandomEngine {.constructor, importcpp: "af::randomEngine(@)", header: "arrayfire.h".}
proc destroy_randomEngine*(this: var RandomEngine) {.importcpp: "#.~randomEngine()", header : "arrayfire.h".}
proc `assign`*(this : RandomEngine, other : RandomEngine)  {.importcpp: "(# = #)", header: "arrayfire.h".}
proc setType*( this : RandomEngine, af_type : RandomEngineType )  {.importcpp: "setType", header : "arrayfire.h".}
proc getType*( this : RandomEngine ) : RandomEngineType {.importcpp: "af::getType", header : "arrayfire.h".}
proc setSeed*( this : RandomEngine, seed : uint64 )  {.importcpp: "setSeed", header : "arrayfire.h".}
proc getSeed*( this : RandomEngine ) : uint64 {.importcpp: "getSeed", header : "arrayfire.h".}
proc get*( this : RandomEngine ) : RandomEngine {.importcpp: "get", header : "arrayfire.h".}

#endregion

