import numpy as np
import Constant as cons
import csv
import json
import rawpy as rp
import Reflectance2XYZ as ref2xyz
import Camera2XYZ as cam2xyz

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
    # print(data[i]['XYZ'])
    data[i]['mLAB'] = ref2xyz.XYZ2LAB(data[i]['XYZ'], 'd65')

#Test
fname = 'RAW_2018_10_07_11_05_37_707.dng'
path = 'E:/UIUC/Data_10_07_18/no_flash/'
labelpath = 'E:/UIUC/Data_10_07_18/Android_label.json'

with open(labelpath) as f:
    label = json.load(f)

out = cam2xyz.rawprocess(path + fname)
acc = 0.0
for i in range(0, 7):
    cc = label[fname][i]
    roi = cc['roi']
    XYZD65 = cam2xyz.getXYZD65(out, roi)
    print(XYZD65)
    ret = MatchColor(XYZD65)
    print(ret)
    if ret == label[fname][i]['label']:
        acc += 1
print(acc / 7)