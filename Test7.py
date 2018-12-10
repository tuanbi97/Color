import colour
import numpy as np
import matplotlib.pyplot as plt
import Reflectance2XYZ as ref2xyz
import Camera2XYZ as cam2xyz
import cv2
import rawpy as rp

def cal_user_wb(wb_roi, raw):
    h, w = np.shape(raw.raw_image)
    print(h, ' ', w)
    # Raw values
    raw_values = np.array(raw.raw_image)
    h, w = np.shape(raw_values)
    rgb = np.zeros([int(h / 2), int(w / 2), 3], dtype=np.float)
    cwb = raw.camera_whitebalance
    print(cwb)
    for i in range(0, np.shape(raw_values)[0], 2):
        for j in range(0, np.shape(raw_values)[1], 2):
            # print(raw.raw_color(i, j), ' ', raw.raw_color(i, j + 1), ' ', raw.raw_color(i + 1, j))
            tmp = [raw.raw_image[i][j + 1], raw.raw_image[i][j], raw.raw_image[i + 1][j]]
            rgb[int(i / 2)][int(j / 2)] = tmp
    vmax = np.max(rgb)
    print(vmax)
    rgb = rgb / vmax
    rgb = rgb * 255
    rgb = rgb.clip(0, 255)

    plt.imshow(rgb.astype(int))
    plt.show()

    # print(raw.raw_colors)
    m_raw_rgb = np.array([0.0, 0.0, 0.0, 0.0])
    m_raw_cnt = np.array([0, 0, 0, 0])
    for i in range(wb_roi[1], wb_roi[3]):
        for j in range(wb_roi[0], wb_roi[2]):
            m_raw_rgb[raw.raw_color(i, j)] += raw.raw_value(i, j)
            m_raw_cnt[raw.raw_color(i, j)] += 1
    m_raw_rgb /= m_raw_cnt
    # print(m_raw_rgb)
    wp = np.array([m_raw_rgb[0], (m_raw_rgb[1] + m_raw_rgb[3]) / 2, m_raw_rgb[2]])
    wp /= wp[1]
    print('white point: ' + str(wp))
    user_whitebalance = (1.0 / wp).tolist() + [0]
    print('user white balance: ' + str(user_whitebalance))
    return user_whitebalance

def prepare_cc():
    print(colour.COLOURCHECKERS.data.keys())
    cc = []
    for i in range(0, 24):
        xyY = colour.COLOURCHECKERS.data['colorchecker 2005'][1].data[i].xyY
        print(colour.COLOURCHECKERS.data['colorchecker 2005'][1].data[i].name)
        print(xyY)
        XYZ = colour.xyY_to_XYZ(xyY)
        Lab = colour.XYZ_to_Lab(XYZ)
        print(
            colour.XYZ_to_sRGB(XYZ, colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'], 'Bradford') * 255)
        # print(colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50'])
        XYZD65 = cam2xyz.XYZD50_XYZD65(XYZ)
        cc.append(XYZD65)
        print(XYZD65, ' ', colour.XYZ_to_sRGB(XYZD65) * 255)
        # print(XYZ, ' ' ,XYZD65)
        # print(colour.RGB_COLOURSPACES['sRGB'].whitepoint, ' ', ill2)
        # print(colour.XYZ_to_sRGB(XYZD65) * 255)
        # print(colour.COLOURCHECKERS.data['cc2005'][1].data[i])
    return cc

cc = prepare_cc()

path = 'E:/UIUC/Data_10_21_18/AndroidMacbeth/RAW_2018_10_21_15_32_27_912_noflash.dng'
path = 'E:/UIUC/Data_11_07_18/Android/RAW_2018_11_09_02_34_34_360_noflash.dng'
path = 'E:/UIUC/Data_11_10_18/android_dng/RAW_2018_11_11_06_33_46_930_noflash.dng'
x, y = (1100, 1174)
box = [[], [], [], [], [], [],
       [], [], [], [], [], [],
       [], [], [], [], [], [],
       [], [], [], [], [], []]
offx = 360
offy = 360
path_jpeg = 'E:/UIUC/Data_11_10_18/android_jpg/JPEG_2018_11_11_06_33_46_930_noflash.jpg'
rgb = cv2.cvtColor(cv2.imread(path_jpeg), cv2.COLOR_BGR2RGB)
for i in range(0, 24):
    xb, yb = (x + (i % 6)* offx, y + int(i / 6) * offy)
    box[i] = [xb, yb, xb + 140, yb + 140]
    rgb = cv2.rectangle(rgb, (box[i][0], box[i][1]), (box[i][2], box[i][3]), color=0, thickness=10)
plt.imshow(rgb)
plt.show()
with rp.imread(path) as raw:
    user_wb = cal_user_wb([1009, 2154, 1140, 2269], raw)
    print(user_wb)
    print(raw.camera_whitebalance)
    custom_xyz = raw.postprocess(user_wb = user_wb,
                                #use_camera_wb = True,
                                 output_bps=16,
                                 output_color=rp.ColorSpace.XYZ,
                                 gamma=(1, 1)).astype(float) / (2.0 ** 16 - 1)
    rgb = raw.postprocess(user_wb = user_wb)
    plt.imshow(rgb)
    plt.show()
    fig, axs = plt.subplots()
    dEs = np.zeros((4, 6), dtype=np.float)
    for i in range(0, 24):
        sample = custom_xyz[box[i][1]:box[i][3], box[i][0]:box[i][2]]
        mean_xyz = [np.mean(sample[:, :, x]) for x in range(0, 3)]
        dE = colour.delta_E(colour.XYZ_to_Lab(mean_xyz, illuminant=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'])
                       , colour.XYZ_to_Lab(cc[i], illuminant=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']))
        #print(dE)
        dEs[int(i / 6)][i % 6] = dE

        # vis = np.tile(mean_xyz, [120, 120, 1])
        # vis2 = np.tile(cc[i], [120, 120, 1])
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # ax1.imshow(sample)
        # ax2.imshow(vis)
        # ax3.imshow(vis2)
        # plt.show()
    axs.imshow(dEs)
    print(np.mean(dEs))
    for i in range(0, 4):
        for j in range(0, 6):
            text = axs.text(j, i, '%.2f' %dEs[i, j],
                           ha="center", va="center", color="w")
    plt.show()
