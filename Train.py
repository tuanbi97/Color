import numpy as np
import json
import Reflectance2XYZ as ref2xyz
import Camera2XYZ as cam2xyz
import matplotlib.pyplot as plt

def distance(a, b):
    return np.sum((np.array(a) - np.array(b))**2)**0.5

def MatchColor(mXYZ):
    best = 1000000000.0
    for i in range(0, len(data)):
        ppg = data[i]['XYZ']
        d = distance(ppg, mXYZ)
        if best > d:
            best = d
            answer = data[i]['name']
    return answer

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
fname = 'RAW_2018_10_07_11_05_37_707.dng'
path = 'E:/UIUC/Data_10_07_18/no_flash/'
labelpath = 'E:/UIUC/Data_10_07_18/Android_label.json'

with open(labelpath) as f:
    label = json.load(f)

out = cam2xyz.rawprocess(path + fname, norm = True)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
# plt.imshow(out)
# plt.show()
wpd50 = [0.964220, 1.000000, 0.825210]
best = 1000000000.0
pos = (0, 0)
for i in range(0, out.shape[0]):
    for j in range(0, out.shape[1]):
        if best > distance(out[i][j], wpd50):
            best = distance(out[i][j], wpd50)
            pos = (i, j)

print (out[pos[0]][pos[1]])
acc = 0.0
for i in range(3, 4):
    print(i, ':')
    cc = label[fname][i]
    roi = cc['roi']
    #visualize D50
    # sample = np.array(out[roi[1]:roi[3], roi[0]:roi[2]], dtype=np.float)
    # for j in range(0, 300):
    #     for k in range(0, 300):
    #         sample[j][k] = cam2xyz.XYZ2RGB(sample[j][k], gamma=1, illuminant='D50')
    # plt.imshow(sample.astype(int))
    # plt.show()

    XYZD65 = cam2xyz.getXYZD65(out, roi)
    #print(cam2xyz.XYZ2RGB(XYZD65, gamma=1, illuminant='D65'))

    #visualize D65
    # for j in range(0, 300):
    #     for k in range(0, 300):
    #         sample[j][k] = cam2xyz.XYZ2RGB(XYZD65, gamma=1, illuminant='D65')
    # plt.imshow(sample.astype(int))
    # plt.show()
    print(XYZD65)
    ret = MatchColor(XYZD65)
    print(ret, '  _  ', label[fname][i]['label'].upper())
    if ret == label[fname][i]['label']:
        acc += 1
print(acc / 7)