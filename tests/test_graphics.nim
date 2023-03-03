import ArrayFireNim
import unittest

test "join":
    setBackend(Backend.AF_BACKEND_CUDA)
    let a = randu(dim4(10,10), Dtype.F32)
    let b = randu(dim4(10,10), Dtype.F32)
    let c = join(1, a, b)
    echo c.dims

# currently also breaks the cpp version
# test "field":
#     setBackend(Backend.AF_BACKEND_DEFAULT) #with 3.4.1 this does not work on CUDA

#     echo $get_available_backends()
#     echo $getactivebackend()

#     let MINIMUM = -3.0
#     let MAXIMUM = 3.0
#     let STEP = 0.18

#     var scale = 2.0

#     window(myWindow, 1024, 1024, "2D Vector Field example")

#     myWindow.grid(1, 2)

#     var dataRange: AFArray = aseq(MINIMUM, MAXIMUM, STEP)

#     var x = tile(dataRange, 1, dataRange.dims(0))
#     var y = tile(dataRange.T, dataRange.dims(0), 1)

#     x.eval()
#     y.eval()

#     let t0 = cpuTime()

#     while not myWindow.close():
#         var points = join(1, flat(x), flat(y))

#         var saddle = join(1, flat(x), -1 * flat(y))

#         var bvals = sin(scale * (x*x + y*y))
#         let t = constant(1, x.elements(), Dtype.F32)
#         let v = flat(bvals)
#         echo v.dtype
#         let hbowl = join(1, v, t)
#         hbowl.eval()

#         echo points.dims

#         myWindow[0, 0].vectorField(points, saddle, "Saddle point")
#         myWindow[0, 1].vectorField(points, hbowl, "hilly bowl (in a loop with varying amplitude)")

#         myWindow.show()

#         scale -= 0.0010
#         if scale < -0.1:
#             scale = 2

#         let time = cpuTime() - t0
#         if time > 0.3:
#             break

#     discard myWindow.close()
