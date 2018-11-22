import numpy as np
import rawpy as rp
import matplotlib.pyplot as plt
import json
import Reflectance2XYZ as ref2xyz

ppg_data = ref2xyz.read_data('VOC 2014 Color Data.csv')

path_im = 'E:/UIUC/Data_11_07_18/iOS'
im_name = 'RAW_2018_11_08_15_09_06_866_noflash.dng'
with open('E:/UIUC/Data_11_07_18/IOSNoFlashLabel.json', 'r') as fp:
    labels = json.load(fp)

key = 'HYDRANGEA FLORET'
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
with rp.imread(path_im + '/' + im_name) as raw:
    patch_roi = labels[key][0]['patch_roi']

    #Raw values
    raw_values = np.array(raw.raw_image)
    w, h = np.shape(raw_values)
    rgb = np.zeros([int(w / 2), int(h / 2), 3], dtype=np.float)
    cwb = raw.camera_whitebalance
    print(cwb)
    for i in range(0, np.shape(raw_values)[0], 2):
        for j in range(0, np.shape(raw_values)[1], 2):
            #print(raw.raw_color(i, j), ' ', raw.raw_color(i, j + 1), ' ', raw.raw_color(i + 1, j))
            tmp = [raw_values[i][j], raw_values[i][j + 1], raw_values[i + 1][j + 1]]
            rgb[int(i / 2)][int(j / 2)] = tmp
    vmax = np.max(rgb)
    print(vmax)
    rgb = rgb / vmax
    rgb = rgb * 255
    rgb = rgb.clip(0, 255)
    ax1.imshow(np.array(rgb[int(patch_roi[1]/2):int(patch_roi[3]/2), int(patch_roi[0]/2):int(patch_roi[2]/2)], np.int))
    ax1.set_title('Raw image')
    ax1.axis('off')

    #white balance
    raw_values = np.array(raw.raw_image)
    w, h = np.shape(raw_values)
    rgb = np.zeros([int(w / 2), int(h / 2), 3], dtype=np.float)
    cwb = raw.camera_whitebalance
    print(cwb)
    for i in range(0, np.shape(raw_values)[0], 2):
        for j in range(0, np.shape(raw_values)[1], 2):
            # print(raw.raw_color(i, j), ' ', raw.raw_color(i, j + 1), ' ', raw.raw_color(i + 1, j))
            tmp = [raw_values[i][j] - 528, raw_values[i][j + 1] - 528, raw_values[i + 1][j + 1] - 528]
            rgb[int(i / 2)][int(j / 2)] = [tmp[0]*cwb[0], tmp[1]*cwb[1], tmp[2]*cwb[2]]
    vmax = np.max(rgb)
    print(vmax)
    rgb = rgb / vmax
    rgb = rgb * 255
    rgb = rgb.clip(0, 255)
    ax2.imshow(np.array(rgb[int(patch_roi[1]/2):int(patch_roi[3]/2), int(patch_roi[0]/2):int(patch_roi[2]/2)], np.int))
    ax2.set_title('White balanced')
    ax2.axis('off')

    #color transform
    rgb = raw.postprocess(use_camera_wb=True, gamma=(1, 1))
    ax3.imshow(rgb[patch_roi[1]:patch_roi[3], patch_roi[0]:patch_roi[2]])
    ax3.set_title('Color space transform')
    ax3.axis('off')

    #full
    rgb = raw.postprocess(use_camera_wb=True)
    ax4.imshow(rgb[patch_roi[1]:patch_roi[3], patch_roi[0]:patch_roi[2]])
    ax4.set_title('Full pipeline')
    ax4.axis('off')

rgb = np.tile(ppg_data[key]['RGB'], [300, 300, 1])
ax5.imshow(rgb)
ax5.set_title('GT')
ax5.axis('off')
plt.show()


