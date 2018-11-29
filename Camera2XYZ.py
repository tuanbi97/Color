import rawpy as rp
import numpy as np
import colour

def XYZD65_XYZD50(pixel):
    M = [[1.0478112, 0.0228866, -0.0501270],
        [0.0295424, 0.9904844, -0.0170491],
        [-0.0092345, 0.0150436, 0.7521316]]
    return [M[0][0] * pixel[0] + M[0][1] * pixel[1] + M[0][2] * pixel[2],
            M[1][0] * pixel[0] + M[1][1] * pixel[1] + M[1][2] * pixel[2],
            M[2][0] * pixel[0] + M[2][1] * pixel[1] + M[2][2] * pixel[2]]

def XYZ2sRGB(pixel, illuminant = 'D65'):
    #print(np.array(colour.XYZ_to_sRGB(pixel) * 255).astype(int))
    if illuminant == 'D65':
        M = [[3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252]]
    else:
        if illuminant == 'D50':
            M = [[3.1338561, -1.6168667, -0.4906146],
                [-0.9787684, 1.9161415, 0.0334540],
                [0.0719453, -0.2289914, 1.4052427]]
    rgb = np.array(
            [M[0][0] * pixel[0] + M[0][1] * pixel[1] + M[0][2] * pixel[2],
            M[1][0] * pixel[0] + M[1][1] * pixel[1] + M[1][2] * pixel[2],
            M[2][0] * pixel[0] + M[2][1] * pixel[1] + M[2][2] * pixel[2]])
    for i in range(0, 3):
        if rgb[i] <= 0.0031308:
            rgb[i] *= 12.92
        else:
            rgb[i] = 1.055 * rgb[i]**(1.0/2.4) - 0.055
    rgb *= 255
    #print(rgb)
    for i in range(0, 3):
        if (rgb[i] > 255 or rgb[i] < 0):
            print("OUT")
            break
    rgb.clip(0, 255)
    #print(rgb)
    return rgb.astype(int)

def RGB2XYZ(pixel, illuminant = 'D65'):
    M = [[0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]]
    pix = np.array(pixel, np.float) / 255.0
    for i in range(0, 3):
        if pix[i] <= 0.04045:
            pix[i] /= 12.92
        else:
            pix[i] = ((pix[i] + 0.055)/1.055) ** 2.4
    xyz = [0.0, 0.0, 0.0]
    for i in range(0, 3):
        for j in range(0, 3):
            xyz[i] += M[i][j] * pix[j]
    return xyz

def XYZ2LAB(xyz, illuminant = 'D65'):
    return colour.XYZ_to_Lab(xyz, illuminant=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'])
    refwhite = [0.9504, 1.0000, 1.0888]
    xyzr = np.array(xyz) / np.array(refwhite)
    e = 0.008856
    k = 903.3
    ff = [0.0, 0.0, 0.0]
    for i in range(0, 3):
        if xyzr[i] > e:
            ff[i] = xyzr[i] ** (1.0/3.0)
        else:
            ff[i] = (k * xyzr[i] + 16)/116
    lab = [0.0, 0.0, 0.0]
    lab[0] = ff[1] * 116 - 16
    lab[1] = 500 * (ff[0] - ff[1])
    lab[2] = 200 * (ff[1] - ff[2])
    return lab

def RGB2LAB(pixel, illuminant = 'D65'):
    xyz = RGB2XYZ(pixel)
    lab = XYZ2LAB(xyz)
    return lab


def getXYZD65(im, roi):
    M = [[0.9555766, -0.0230393, 0.0631636],
        [-0.0282895, 1.0099416, 0.0210077],
        [0.0122982, -0.0204830, 1.3299098]]

    # M = [[1.0, 1.0, 1.0],
    #      [1.0, 1.0, 1.0],
    #      [1.0, 1.0, 1.0]]

    X = Y = Z = 0.0
    for i in range(roi[1], roi[3]):
        for j in range(roi[0], roi[2]):
            X += im[i][j][0]
            Y += im[i][j][1]
            Z += im[i][j][2]
    s = (roi[3] - roi[1])*(roi[2] - roi[0])
    X /= s
    Y /= s
    Z /= s
    return [M[0][0] * X + M[0][1] * Y + M[0][2] * Z,
            M[1][0] * X + M[1][1] * Y + M[1][2] * Z,
            M[2][0] * X + M[2][1] * Y + M[2][2] * Z]

def getXYZD50(im, roi):

    X = Y = Z = 0.0
    for i in range(roi[1], roi[3]):
        for j in range(roi[0], roi[2]):
            X += im[i][j][0]
            Y += im[i][j][1]
            Z += im[i][j][2]
    s = (roi[3] - roi[1])*(roi[2] - roi[0])
    X /= s
    Y /= s
    Z /= s
    return [X, Y, Z]

def rawprocess(fpath, norm = False):
    with rp.imread(fpath) as raw:
        #user_wb = [1.868109862326931, 1.0, 1.4500039644365266, 0]
        #user_wb = [1.9026140427049316, 1.0, 1.7919425716057134, 0]
        user_wb = raw.camera_whitebalance
        rgb = raw.postprocess(demosaic_algorithm=rp.DemosaicAlgorithm.LINEAR,
                              half_size=False,
                              four_color_rgb=False,
                              use_camera_wb=False,
                              use_auto_wb=False,
                              user_wb=user_wb,
                              output_color=rp.ColorSpace.XYZ,
                              output_bps=8,
                              no_auto_bright=True,
                              gamma=(1, 1))
        if norm == False:
            return rgb
        else:
            return rgb / 255.0
