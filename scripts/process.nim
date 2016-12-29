import os
import strutils

let headers = ["version.h","defines.h","array_clean.h","backend.h","arith.h","blas.h","complex_clean.h",
                "data.h","device.h","features.h","gfor.h","graphics.h","image.h","index.h", "lapack.h",
                "cuda.h","opencl.h","random.h","seq.h","signal.h","sparse.h","statistics.h","util.h",
                "version.h","vision.h","array_proxy_clean.h","algorithm.h","timing.h"]

for header in headers:
  var (_,basename,_) = splitFile(header)
  echo "processing header: $1"%header
  let cmd = "c2nim --cpp include/arrayfire.c2nim include/$1 --out:processed/$2" % [header,basename&".nim"]
  if os.execShellCmd(cmd) == 0:
    echo "c2nim of $1 ok"%header
  else:
    echo "c2nim of $1 failed"%header