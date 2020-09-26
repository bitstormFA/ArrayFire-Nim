import unittest
import strutils
import ArrayFireNim
import math
import os


suite "benchmarks":
  test "blas":
    setDevice(0)
    info()

    echo "Benchmark N-by-N matrix multiply\n"

    var peak : float = 0

    for n in countup(128,2048,128):
      stdout.write "$1 x $2 : " % [$n,$n]

      let a = constant(1,n,n,f32)
      var time : float
      timeit(time): 
        var b = matmul(a,a)
        b.eval()

      let gflops = float(2 * n^3) / (float(time) * 1.0e9)

      if gflops > peak:
        peak = gflops  
      echo "$1 Gflops" % gflops.formatFloat(precision=4)

    echo "### peak $1 Gflops\n" % peak.formatFloat(precision=4)

  test "fft":
    setDevice(0)
    info()
    echo "Benchmarking N-by-N 2D fft"

    for m in 7..12:
      var n = (1 shl m)

      stdout.write "$1 x $2 : " % [$n,$n]
      var a = randu(n,n)

      var time : float

      timeit(time):
        var b = fft2(a)
        b.eval()

      let gflops = float(10 * n * n * m) / (time * 1e9)

      echo "$1 Gflops" % gflops.formatFloat(precision=4)


