import numpy as np
import json
import Reflectance2XYZ as ref2xyz
import Camera2XYZ as cam2xyz
import matplotlib.pyplot as plt
import glob
import cv2

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

data = ref2xyz.read_data('VOC 2014 Color Data.csv')
print(len(data))
for key in data.keys():
    data[key]['XYZ'] = ref2xyz.reflectance2XYZ(data[key]['ref'])
    data[key]['XYZD50'] = cam2xyz.XYZD65_XYZD50(data[key]['XYZ'])

#Test
#fname = 'RAW_2018_10_7_11_33_16_820.dng'
path = 'E:/UIUC/Data_10_21_18/Android'
path_images = glob.glob(path + '/*_noflash.jpg')
labelpath = 'E:/UIUC/Data_10_21_18/AndroidNoFlashLabel.json'

for pim in path_images:
    im = cv2.imread(pim)
    im_name = pim[len(path) + 1:]
    print(im_name)
    with open(labelpath) as f:
        label = json.load(f)
    print(label[im_name][0]['label'])
    # h, w = (np.shape(im)[0], np.shape(im)[1])
    # sim = cv2.resize(im, (int(w / 4), int(h / 4)))
    # cv2.imshow('im', sim)
    # cv2.waitKey(0)

    roi = [label[im_name][0]['x1'], label[im_name][0]['y1'], label[im_name][0]['x2'], label[im_name][0]['y2']]
    sample = cv2.cvtColor(im[roi[1]:roi[3],roi[0]:roi[2]], cv2.COLOR_BGR2RGB)

    plt.imshow(sample)
    plt.show()
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
    plt.imshow(vis_s)
    plt.show()
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