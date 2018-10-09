import numpy as np
import json
import Reflectance2XYZ as ref2xyz
import Camera2XYZ as cam2xyz
import matplotlib.pyplot as plt

def distance(a, b):
    return np.sum((np.array(a) - np.array(b))**2)**0.5

def cmp(x):
    return x[0]

def MatchColor(mXYZ):
    best = 1000000000.0
    ret = []
    for i in range(0, len(data)):
        ppg = data[i]['XYZ']
        d = distance(ppg, mXYZ)
        ret.append((d, data[i]['name'], data[i]['RGB']))
        if best > d:
            best = d
            answer = data[i]['name']
    ret = sorted(ret, key=cmp)
    return answer, ret

data = ref2xyz.read_data('VOC 2014 Color Data.csv')
print(len(data))
for i in range(0, len(data)):
    data[i]['XYZ'] = ref2xyz.reflectance2XYZ(data[i]['ref'])
    #check visualization
    if data[i]['name'] == 'RIVER ROUGE':
        xyz = data[i]['XYZ']
        rgb = cam2xyz.XYZ2RGB(xyz, gamma=2.2, illuminant='D65')
        print(rgb,' ', data[i]['RGB'], ' ', data[i]['XYZ'])
        # sample = np.tile([int(x) for x in rgb], (300, 300, 1))
        # plt.imshow(sample)
        # plt.show()
        #print(data[i]['XYZ'])
    #print(data[i]['XYZ'])
    data[i]['mLAB'] = ref2xyz.XYZ2LAB(data[i]['XYZ'], 'd65')

#Test
fname = 'RAW_2018_10_7_11_33_16_820.dng'
path = 'E:/UIUC/Data_10_07_18/iOS/no_flash/'
labelpath = 'E:/UIUC/Data_10_07_18/IOSNoFlashLabel.json'

with open(labelpath) as f:
    label = json.load(f)

out = cam2xyz.rawprocess(path + fname, norm = True)
# fig_size = plt.rcParams["figure.figsize"]
# fig_size[0] = 12
# fig_size[1] = 9
# plt.rcParams["figure.figsize"] = fig_size
# plt.imshow(out)
# plt.show()
# wpd50 = [0.964220, 1.000000, 0.825210]
# best = 1000000000.0
# pos = (0, 0)
# for i in range(0, out.shape[0]):
#     for j in range(0, out.shape[1]):
#         if best > distance(out[i][j], wpd50):
#             best = distance(out[i][j], wpd50)
#             pos = (i, j)

#print (out[pos[0]][pos[1]])
acc = 0.0
for i in range(0, 7):
    print(i, ':')
    cc = label[fname][i]
    roi = cc['roi']
    #visualize D50
    sample = np.array(out[roi[1]:roi[3], roi[0]:roi[2]], dtype=np.float)
    for j in range(0, 300):
        for k in range(0, 300):
            sample[j][k] = cam2xyz.XYZ2RGB(sample[j][k], gamma=2.22, illuminant='D50')
    plt.imshow(sample.astype(int))
    plt.show()

    XYZD50, XYZD65 = cam2xyz.getXYZD65(out, roi)
    print(cam2xyz.XYZ2RGB(XYZD65, gamma=2.2, illuminant='D65'))

    print('XYZ D50', XYZD50)
    print('XYZ D65', XYZD65)
    ret, top_k = MatchColor(XYZD65)
    print(ret, '  _  ', label[fname][i]['label'].upper())
    # visualize D65
    rgbd65 = np.tile(cam2xyz.XYZ2RGB(XYZD65, gamma=2.22, illuminant='D65'), (300, 300, 1))
    # plt.imshow(rgbd65.astype(int))
    # plt.show()
    for j in range(0, 10):
        print(top_k[j][1], ':', top_k[j][0])
        sample = np.tile(top_k[j][2], (300, 300, 1))
        f, ax = plt.subplots(1, 2)
        ax[0].imshow(rgbd65.astype(int))
        ax[0].set_title('RGB D65')
        ax[1].imshow(sample)
        ax[1].set_title(top_k[j][1])
        plt.show()
    if ret == label[fname][i]['label']:
        acc += 1
print(acc / 7)