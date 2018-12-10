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

test_key = ['HYDRANGEA FLORET', 'BLUE SHAMROCK', 'DARK GREEN VELVET', 'DOVER GRAY', 'OREGANO', 'GINGER', 'FERRIS WHEEL']

fig, axs = plt.subplots(len(test_key), 5)
cnt = 0
for key in test_key:
    im_name = labels[key][0]['im_name']
    with rp.imread(path_im + '/' + im_name) as raw:
        patch_roi = labels[key][0]['patch_roi']

        #Raw values
        raw_values = np.array(raw.raw_image)
        h, w = np.shape(raw_values)
        rgb = np.zeros([int(h / 2), int(w / 2), 3], dtype=np.float)
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
        axs[cnt][0].imshow(np.array(rgb[int(patch_roi[1]/2):int(patch_roi[3]/2), int(patch_roi[0]/2):int(patch_roi[2]/2)], np.int))
        if (cnt == 0):
            axs[cnt][0].set_title('Raw image')
        axs[cnt][0].tick_params(axis='x', which='both', bottom=False, top = False, labelbottom=False)
        axs[cnt][0].tick_params(axis='y', which='both', left=False, right = False, labelleft=False)
        # ax1.xticks([])
        # ax1.yticks([])
        #ax1.axis('off')

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
        axs[cnt][1].imshow(np.array(rgb[int(patch_roi[1]/2):int(patch_roi[3]/2), int(patch_roi[0]/2):int(patch_roi[2]/2)], np.int))
        if (cnt == 0):
            axs[cnt][1].set_title('White balanced')
        axs[cnt][1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[cnt][1].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        #color transform
        rgb = raw.postprocess(use_camera_wb=True, gamma=(1, 1))
        axs[cnt][2].imshow(rgb[patch_roi[1]:patch_roi[3], patch_roi[0]:patch_roi[2]])
        if (cnt == 0):
            axs[cnt][2].set_title('Color space transform')
        axs[cnt][2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[cnt][2].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        #full
        rgb = raw.postprocess(use_camera_wb=True)
        axs[cnt][3].imshow(rgb[patch_roi[1]:patch_roi[3], patch_roi[0]:patch_roi[2]])
        if (cnt == 0):
            axs[cnt][3].set_title('Tone mapping')
        axs[cnt][3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axs[cnt][3].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    rgb = np.tile(ppg_data[key]['RGB'], [300, 300, 1])
    axs[cnt][4].imshow(rgb)
    if (cnt == 0):
        axs[cnt][4].set_title('Ground truth')
    axs[cnt][4].tick_params(axis='x', which='both', bottom=False, top = False, labelbottom=False)
    axs[cnt][4].tick_params(axis='y', which='both', left=False, right = False, labelleft=False)
    cnt += 1
plt.show()


