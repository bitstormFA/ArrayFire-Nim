import ArrayFire_Nim
import unittest
import strutils
import std/times

test "fast":
    setDevice(0)
    info()
    var img_color = loadImage("assets/man.jpg", true)
    let img = colorSpace(img_color, CspaceT.AF_GRAY, CSpaceT.AF_RGB)

    img_color /= 255

    let feat = fast(img, 20.0, 9, true, 0.05, 3)

    let hx = feat.getX().to_seq(float)
    let hy = feat.getY().to_seq(float)

    let draw_len = 3

    for f in 0..<feat.getNumFeatures():
        let x = int(hx[f])
        let y = int(hy[f])
        img_color[y, aseq(int(x)-draw_len, x+draw_len), 0] = 0
        img_color[y, aseq(int(x)-draw_len, x+draw_len), 1] = 1
        img_color[y, aseq(int(x)-draw_len, x+draw_len), 2] = 0

        img_color[aseq(y-draw_len, y+draw_len), x, 0] = 0
        img_color[aseq(y-draw_len, y+draw_len), x, 1] = 1
        img_color[aseq(y-draw_len, y+draw_len), x, 2] = 0

    echo "Features found: $1" % $feat.getNumFeatures

    var wnd: Window

    wnd.set_title("FAST Feature Detector")

    let t0 = cpuTime()

    while not wnd.close():
        wnd.image(img_color, "FAST Feature Detector")
        let time = cpuTime()-t0
        if time > 0.05:
            break

    discard wnd.close()

test "harris":
    setDevice(0)
    info()

    var img_color = loadImage("assets/man.jpg", true)
    let img = colorSpace(img_color, CSpaceT.AF_GRAY, CSpaceT.AF_RGB)

    img_color /= 255.0

    var (ix, iy) = img.grad()


    var ixx = ix * ix
    var ixy = ix * iy
    var iyy = iy * iy

    var gauss_filt = gaussianKernel(5, 5, 1.0, 1.0)

    ixx = ixx.convolve(gauss_filt, ConvMode.AF_CONV_DEFAULT, ConvDomain.AF_CONV_AUTO)
    ixy = ixy.convolve(gauss_filt, ConvMode.AF_CONV_DEFAULT, ConvDomain.AF_CONV_AUTO)
    iyy = iyy.convolve(gauss_filt, ConvMode.AF_CONV_DEFAULT, ConvDomain.AF_CONV_AUTO)

    var itr = ixx + iyy

    var idet = ixx * iyy - ixy * ixy

    var response = idet - 0.04 * (itr * itr)

    var mask = constant(1, dim4(3, 3), ty = Dtype.F32)

    var max_resp = dilate(response, mask)

    var corners = response > 1e5
    corners = corners * response

    corners = (corners == max_resp) * corners

    var h_corners = corners.to_seq(float)

    var good_corners: cuint = 0

    let draw_len: int = 3

    for y in draw_len..<img_color.dims(0)-draw_len:
        for x in draw_len..<img_color.dims(1)-draw_len:
            if h_corners[int(x * corners.dims(0) + y)] > 1e5:

                img_color[y, aseq(x-draw_len, x+draw_len), 0] = 0
                img_color[y, aseq(x-draw_len, x+draw_len), 1] = 1
                img_color[y, aseq(x-draw_len, x+draw_len), 2] = 0

                img_color[aseq(y-draw_len, y+draw_len), x, 0] = 0;
                img_color[aseq(y-draw_len, y+draw_len), x, 1] = 1;
                img_color[aseq(y-draw_len, y+draw_len), x, 2] = 0;

                good_corners+=1


    echo "Corners found: $1" % $good_corners
    window(wnd2, "Harris Corner Detector")

    let t0 = cpuTime()

    while not wnd2.close():
        wnd2.image(img_color, "ArrayFire")
        let time = cpuTime()-t0
        if time > 0.05:
            break

    discard wnd2.close()
