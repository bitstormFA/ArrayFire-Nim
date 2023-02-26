import ArrayFire_Nim
import unittest

test "svd":
    let ain = afa(2,3, @[1, 4, 2, 5, 3, 6], Dtype.F32)
    let (u, s, v) = svd(ain)
    let s_mat = diag(s, 0, false)
    let in_recon = matmul(u, s_mat, v[aseq(2), span])

    # echo ain 
    # echo s
    # echo u 
    # echo s_mat 
    # echo v
    # echo in_recon

    check in_recon.to_seq(int) == @[1, 4, 2, 5, 3, 6] 