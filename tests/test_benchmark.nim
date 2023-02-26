import ArrayFire_Nim
import unittest
import strutils
import strformat
import std/math

test "benchmark blas":
    setDevice(0)
    info()

    echo "Benchmark N-by-N matrix multiply\n"

    var peak: float = 0

    for n in countup(128, 2048, 128):
        stdout.write fmt"{n:>5} x {n:>5}: "

        let a = constant(1, n, n, Dtype.F32)
        let time = timeit:
            var b = matmul(a, a)
            b.eval()

        let ops = float(2 * n ^ 3)
        let gflops = ops / (time * 1e9)

        if gflops > peak:
            peak = gflops
        echo fmt"{gflops:>8.0f}"

    echo fmt"### peak {peak:8} Gflops\n" 

test "benchmark fft":
    setDevice(0)
    info()
    echo "Benchmarking N-by-N 2D FFT"

    for m in 7..12:
        var n = (1 shl m)

        stdout.write fmt"{n:>5} x {n:>5}: "
        let a = randu(n, n, Dtype.F32)

        let time = timeit:
            let b = fft2(a)
            b.eval()

        let gflops = float64(10 * n * n * m) / (time * 1e9)

        echo fmt"{gflops:>8.0f} Gflops"
