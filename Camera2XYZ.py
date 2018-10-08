import rawpy as rp

def getXYZD65(im, roi):
    M = [[0.9555766, -0.0230393, 0.0631636],
                [-0.0282895, 1.0099416, 0.0210077],
                [0.0122982, -0.0204830, 1.3299098]]
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

def rawprocess(fpath):
    with rp.imread(fpath) as raw:
        user_wb = [1.868109862326931, 1.0, 1.4500039644365266, 0]
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
        return rgb
