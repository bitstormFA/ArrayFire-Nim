import ArrayFireNim
# this is meant to be run with valgrind

proc alloc_and_assign() =
    var a = randu(10000,10000, Dtype.F64)
    let view = a[aseq(0, 200), aseq(200,300)]
    var v2 = view * view 
    v2 *= cdouble(1.5)
    a += 2i32
    let b = a * 2i32
    let c = b + a 
    c.eval()


set_backend(Backend.AF_BACKEND_CPU)
alloc_and_assign()
    
