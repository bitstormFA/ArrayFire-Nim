import unittest
import strutils
import ArrayFireNim
import os
import times


suite "graphics":
  test "field":
    #setBackend(Backend.CPU) #with 3.4.1 this does not work on CUDA

    echo $get_available_backends()
    echo $getactivebackend()

    let MINIMUM = -3'f32
    let MAXIMUM = 3'f32
    let STEP = 0.18'f32

    var scale = 2'f32

    window(myWindow, 1024, 1024, "2D Vector Field example")

    myWindow.grid(1,2)

    myWindow[0,0].setAxesLimits(MINIMUM,MAXIMUM,MINIMUM,MAXIMUM)
    myWindow[0,1].setAxesLimits(MINIMUM,MAXIMUM,MINIMUM,MAXIMUM)

    var dataRange : Matrix = mseq(MINIMUM,MAXIMUM,STEP)

    var x = tile(dataRange, 1, dataRange.dims(0))
    var y = tile(dataRange.T, dataRange.dims(0), 1)

    x.eval()
    y.eval()

    let t0 = cpuTime()

    while not myWindow.close():
      var points = join(1, flat(x), flat(y))

      var saddle = join(1, flat(x), -1'f32 * flat(y))

      var bvals = sin( scale * ( x*x + y*y ) )
      var hbowl = join(1, constant(1i32, x.elements()), flat(bvals))
      hbowl.eval()

      myWindow[0, 0].vectorField(points, saddle, "Saddle point")
      myWindow[0, 1].vectorField(points, hbowl, "hilly bowl (in a loop with varying amplitude)")
      myWindow.show()

      scale += 0.0010'f32
      if scale < -0.1'f32:
        scale = 2'f32

      let time = cpuTime() - t0
      if time > 2:
        break




