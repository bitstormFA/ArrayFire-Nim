import unittest
import ArrayFireNim


suite "linear algebra":
  test "svd":
    setDevice(0)
    info()
    
    var ain = afa(2,3,@[1'f32, 4'f32, 2'f32, 5'f32, 3'f32, 6'f32])
    var (u,s_vec,vt) = ain.svd()

    var s_mat = diag(s_vec ,0, false)
    var in_recon = matmul(u,s_mat, vt[mseq(2), span])

    echo $ain
    echo $s_vec
    echo $u
    echo $s_mat
    echo $vt
    echo $in_recon