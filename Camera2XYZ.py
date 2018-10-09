import rawpy as rp
import numpy as np

def XYZD65_XYZD50(pixel):
    M = [[1.0478112, 0.0228866, -0.0501270],
        [0.0295424, 0.9904844, -0.0170491],
        [-0.0092345, 0.0150436, 0.7521316]]
    return [M[0][0] * pixel[0] + M[0][1] * pixel[1] + M[0][2] * pixel[2],
            M[1][0] * pixel[0] + M[1][1] * pixel[1] + M[1][2] * pixel[2],
            M[2][0] * pixel[0] + M[2][1] * pixel[1] + M[2][2] * pixel[2]]

def XYZ2RGB(pixel, gamma = 2.2, illuminant = 'D50'):
    if illuminant == 'D65':
        M = [[3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252]]
    else:
        if illuminant == 'D50':
            M = [[3.1338561, -1.6168667, -0.4906146],
                [-0.9787684, 1.9161415, 0.0334540],
                [0.0719453, -0.2289914, 1.4052427]]
    rgb = np.array([M[0][0] * pixel[0] + M[0][1] * pixel[1] + M[0][2] * pixel[2],
            M[1][0] * pixel[0] + M[1][1] * pixel[1] + M[1][2] * pixel[2],
            M[2][0] * pixel[0] + M[2][1] * pixel[1] + M[2][2] * pixel[2]])
    rgb = rgb.clip(0, 1)
    rgb = rgb ** (1.0/gamma)
    rgb *= 255
    rgb = rgb.clip(0, 255)
    return rgb.astype(int)

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
