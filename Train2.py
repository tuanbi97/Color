import numpy as np
import json
import Reflectance2XYZ as ref2xyz
import Camera2XYZ as cam2xyz
import matplotlib.pyplot as plt
import glob
import cv2
import rawpy as rp

def distance(a, b):
    return np.sum((np.array(a) - np.array(b))**2)**0.5

def MatchColor(mXYZ):
    best = 1000000000.0
    for i in range(0, len(data)):
        ppg = data[i]['XYZD50']
        d = distance(ppg, mXYZ)
        if best > d:
            best = d
            answer = data[i]['name']
    return answer

def deltaE(lab1, lab2):
    return np.sum((np.array(lab1) - np.array(lab2)) ** 2) ** 0.5

data = ref2xyz.read_data('VOC 2014 Color Data.csv')
print(len(data))
for key in data.keys():
    data[key]['XYZ'] = ref2xyz.reflectance2XYZ(data[key]['ref'])
    data[key]['XYZD50'] = cam2xyz.XYZD65_XYZD50(data[key]['XYZ'])

def array2str(arr):
    return (str(arr[0]) + ',' + str(arr[1]) + ',' + str(arr[2]))

#Test
#fname = 'RAW_2018_10_7_11_33_16_820.dng'
path = 'E:/UIUC/Data_10_21_18/Android'
#path = 'E:/UIUC/Data_10_21_18/AndroidDNGprocessed'
path_images = glob.glob(path + '/*_noflash.dng')
labelpath = 'E:/UIUC/Data_10_21_18/AndroidNoFlashLabel.json'
Es = []

exceptions = [
'E:/UIUC/Data_10_21_18/Android\RAW_2018_10_21_14_56_33_317_noflash.dng',
'E:/UIUC/Data_10_21_18/Android\RAW_2018_10_21_15_06_14_476_noflash.dng',
'E:/UIUC/Data_10_21_18/Android\RAW_2018_10_21_15_12_29_434_noflash.dng',
'E:/UIUC/Data_10_21_18/Android\RAW_2018_10_21_14_37_52_141_noflash.dng'
]

# exceptions = [
# 'E:/UIUC/Data_10_21_18/AndroidDNGprocessed\RAW_2018_10_21_14_56_33_317_noflash.jpg',
# 'E:/UIUC/Data_10_21_18/AndroidDNGprocessed\RAW_2018_10_21_15_06_14_476_noflash.jpg',
# 'E:/UIUC/Data_10_21_18/AndroidDNGprocessed\RAW_2018_10_21_15_12_29_434_noflash.jpg',
# 'E:/UIUC/Data_10_21_18/AndroidDNGprocessed\RAW_2018_10_21_14_37_52_141_noflash.jpg'
# ]

file = open('colors.csv', 'w')
file.write('Label, PPG_Red, PPG_Green, PPG_Blue, PPG_L, PPG_A, PPG_B, JPG_R, JPG_G, JPG_B, JPG_L, JPG_A, JPG_B\n')
for pim in path_images:
    print(pim)
    if pim in exceptions:
        continue
    with rp.imread(pim) as raw:
        im = raw.postprocess(user_wb = [1.9026140427049316, 1.0, 1.7919425716057134, 0])
    #im = cv2.imread(pim)
    im_name = 'JPEG' + pim[len(path) + 4:-4] + '.jpg'
    # im = cv2.imread(path + '/' + im_name)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # print(im_name)
    with open(labelpath) as f:
        label = json.load(f)
    print(label[im_name][0]['label'].upper())
    # h, w = (np.shape(im)[0], np.shape(im)[1])
    # sim = cv2.resize(im, (int(w / 4), int(h / 4)))
    # cv2.imshow('im', sim)
    # cv2.waitKey(0)

    roi = [label[im_name][0]['x1'], label[im_name][0]['y1'], label[im_name][0]['x2'], label[im_name][0]['y2']]
    sample = im[roi[1]:roi[3],roi[0]:roi[2]]

    # plt.imshow(sample)
    # plt.show()
    mr = mg = mb = 0.0
    for i in range(0, np.shape(sample)[0]):
        for j in range(0, np.shape(sample)[1]):
            mr += sample[i][j][0]
            mg += sample[i][j][1]
            mb += sample[i][j][2]
    ss = np.shape(sample)[0] * np.shape(sample)[1]
    mr /= ss
    mg /= ss
    mb /= ss
    vis_s = np.tile([int(mr), int(mg), int(mb)], [300, 300, 1])
    # plt.imshow(vis_s)
    # plt.show()
    c_rgb = [int(mr), int(mg), int(mb)]
    print('dng rgb:')
    print(c_rgb)
    clabel = label[im_name][0]['label'].upper()
    print('lab rgb')
    print(cam2xyz.RGB2LAB(c_rgb))
    dE = 0.0
    if clabel in data.keys():
        print('ppg lab')
        print(data[clabel]['LAB'])
        print(deltaE(cam2xyz.RGB2LAB(c_rgb), data[clabel]['LAB']))
        dE = deltaE(cam2xyz.RGB2LAB(c_rgb), data[clabel]['LAB'])
        Es.append(deltaE(cam2xyz.RGB2LAB(c_rgb), data[clabel]['LAB']))
        print('ppg rgb')
        print(data[clabel]['RGB'])
        file.write(clabel + ',' + array2str(data[clabel]['RGB']) + ',' + array2str(data[clabel]['LAB']) + ',' + array2str(c_rgb) + ',' + array2str(cam2xyz.RGB2LAB(c_rgb)) + '\n')

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle('delta E = ' + str(dE), fontsize=16)
    # ax1.set_title('Processed DNG')
    # ax1.imshow(vis_s)
    # if clabel in data.keys():
    #     ax2.set_title('PPG')
    #     ax2.imshow(np.tile(data[clabel]['RGB'], [300, 300, 1]))
    # plt.show()
    #if clabel == 'RIVER ROUGE':


    #c_lab = cam2xyz.RGB2LAB(c_rgb)
    # plt.imshow(sample)
    # plt.show()

    # #out = cam2xyz.rawprocess(path + fname, norm = True)
    # fig_size = plt.rcParams["figure.figsize"]
    # fig_size[0] = 12
    # fig_size[1] = 9
    # plt.rcParams["figure.figsize"] = fig_size
    # # plt.imshow(out)
    # # plt.show()
    # # maxln = 0.0
    # # for i in range(0, out.shape[0]):
    # #     for j in range(0, out.shape[1]):
    # #         maxln = max(maxln, out[i][j][1])
    #
    # # out = out.astype(float) / 0.07058824
    #
    # acc = 0.0
    # for i in range(0, 7):
    #     print(i, ':')
    #     cc = label[fname][i]
    #     roi = cc['roi']
    #     #visualize D50
    #     sample = np.array(out[roi[1]:roi[3], roi[0]:roi[2]], dtype=np.float)
    #     for j in range(0, 300):
    #         for k in range(0, 300):
    #             sample[j][k] = cam2xyz.XYZ2RGB(sample[j][k], gamma=2.22, illuminant='D50')
    #     plt.imshow(sample.astype(int))
    #     plt.show()
    #
    #     XYZD50 = cam2xyz.getXYZD50(out, roi)
    #     ret = MatchColor(XYZD50)
    #     print(ret, '  _  ', label[fname][i]['label'].upper())
    #     if ret == label[fname][i]['label']:
    #         acc += 1
    # print(acc / 7)
file.close()
print(np.mean(Es))
print(np.max(Es))
print(np.min(Es))