import numpy as np
import colour
import Reflectance2XYZ as ref2XYZ
import Camera2XYZ as cam2xyz
import matplotlib.pyplot as plt

def Distance(xyz, lab):
    lab1 = colour.XYZ_to_Lab(xyz, colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'])
    return colour.delta_E(lab, lab1)
def comp(item):
    return item[0]

ppg_data = ref2XYZ.read_data('VOC 2014 Color Data.csv')
ra = {}
with open('ResultAndroid.txt', 'r') as rf:
    n = int(rf.readline())
    lines = rf.readlines()
    for i in range(0, int(n)):
        line = lines[i].split(',')
        color_name = line[0].replace('_', ' ')
        ra[color_name] = {}
        ra[color_name]['jRGB'] = [int(x) for x in line[4:7]]
        ra[color_name]['dXYZ'] = [float(x) for x in line[7:10]]
        ra[color_name]['cXYZ'] = [float(x) for x in line[10:13]]
check = 'jRGB'
top = np.zeros(10, dtype=np.float)
for c1 in ra.keys():
    dc = []
    print(c1)
    for c2 in ppg_data.keys():
        if c2 in ra.keys():
            dc.append((Distance(cam2xyz.RGB2XYZ(ra[c1][check]), ppg_data[c2]['LAB']), c2))
    dc = sorted(dc, key=comp)
    for i in range(0, len(top)):
        if c1 in [x[1] for x in dc[0:i + 1]]:
            top[i] += 1.0

    # print(dc)
    # for i in range(0, 10):
    #     rgb1 = ppg_data[dc[i][1]]['RGB']
    #     rgb2 = colour.XYZ_to_sRGB(ra[c1][check])
    #     vis1 = np.tile(rgb1, [300, 300, 1])
    #     vis2 = np.tile(rgb2, [300, 300, 1])
    #     fig, (ax1, ax2) = plt.subplots(1, 2)
    #     ax1.set_title(dc[i][1])
    #     ax1.imshow(vis1)
    #     ax2.set_title(c1)
    #     ax2.imshow(vis2)
    #     plt.show()
    # break
print(top)
print(top / n)