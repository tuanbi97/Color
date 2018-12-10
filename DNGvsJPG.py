import rawpy as rp
import numpy as np
import cv2
import matplotlib.pyplot as plt
import Camera2XYZ as cam2xyz
import Reflectance2XYZ as ref2xyz
#import exiftool
import glob
import shutil
import os
import json
import colour

def test_color(path_im, color_name, ppg_data, wb_roi, patch_roi, path_jpeg = None):
    #Full DNG detail
    # with exiftool.ExifTool() as et:
    #     metadata = et.get_metadata(path_im)
    #     for key in metadata.keys():
    #         print(key, ' ', metadata[key])

    with rp.imread(path_im) as raw:
        h, w = np.shape(raw.raw_image)
        print(h, ' ', w)


        #print(raw.raw_colors)
        m_raw_rgb = np.array([0.0, 0.0, 0.0, 0.0])
        m_raw_cnt = np.array([0, 0, 0, 0])
        for i in range(wb_roi[1], wb_roi[3]):
            for j in range(wb_roi[0], wb_roi[2]):
                m_raw_rgb[raw.raw_color(i, j)] += raw.raw_value(i, j)
                m_raw_cnt[raw.raw_color(i, j)] += 1
        m_raw_rgb /= m_raw_cnt
        # print(m_raw_rgb)
        wp = np.array([m_raw_rgb[0], m_raw_rgb[1], m_raw_rgb[2]])
        wp /= wp[1]
        print('white point: ' + str(wp))
        user_whitebalance = (1.0 / wp).tolist() + [0]
        print('user white balance: ' + str(user_whitebalance))
        print('camera white balance: ' + str(raw.camera_whitebalance))

        print('Custom processing:')
        custom_xyz = raw.postprocess(user_wb=user_whitebalance,
                                     output_bps=16,
                                     output_color=rp.ColorSpace.XYZ,
                                     gamma = (1, 1)).astype(float) / (2.0**16 - 1)

        sample = custom_xyz[patch_roi[1]:patch_roi[3], patch_roi[0]:patch_roi[2]]
        c_xyz = [np.mean(sample[:, :, x]) for x in range(0, 3)]
        c_rgb = cam2xyz.XYZ2sRGB(c_xyz)

        print('Default processing:')
        default_xyz = raw.postprocess(use_camera_wb=True,
                                     output_bps=16,
                                     output_color=rp.ColorSpace.XYZ,
                                     gamma = (1, 1)).astype(float) / (2.0**16 - 1)
        sample = default_xyz[patch_roi[1]:patch_roi[3], patch_roi[0]:patch_roi[2]]
        d_xyz = [np.mean(sample[:, :, x]) for x in range(0, 3)]
        d_rgb = cam2xyz.XYZ2sRGB(d_xyz)
        j_rgb = [0, 0, 0]
        if path_jpeg:
            jpeg_rgb = cv2.cvtColor(cv2.imread(path_jpeg), cv2.COLOR_BGR2RGB)
            sample = jpeg_rgb[patch_roi[1]:patch_roi[3], patch_roi[0]:patch_roi[2]]
            j_rgb = [np.mean(sample[:, :, x]) for x in range(0, 3)]

        c_lab = cam2xyz.XYZ2LAB(c_xyz)
        d_lab = cam2xyz.XYZ2LAB(d_xyz)
        if path_jpeg:
            j_lab = cam2xyz.RGB2LAB(j_rgb)
        print('RGB after eliminate illuminant: ' + str(c_rgb))
        print('PPG RGB: ' + str(ppg_data[color_name]['RGB']))
        print('Lab after eliminate illuminant: ' + str(c_lab))
        print('PPG Lab: ' + str(ppg_data[color_name]['LAB']))

        # dE_c = deltaE(c_lab, np.array(ppg_data[color_name]['LAB']))
        # dE_d = deltaE(d_lab, np.array(ppg_data[color_name]['LAB']))
        dE_c = colour.delta_E(c_lab, np.array(ppg_data[color_name]['LAB']))
        dE_d = colour.delta_E(d_lab, np.array(ppg_data[color_name]['LAB']))
        # print('Custom wb dE: ', deltaE(c_lab, np.array(ppg_data[color_name]['LAB'])))
        # print('Default wb dE: ', deltaE(d_lab, np.array(ppg_data[color_name]['LAB'])))
        print('Custom wb dE: ', dE_c)
        print('Default wb dE: ', dE_d)


        dE_j = -1.0
        if path_jpeg:
            dE_j = colour.delta_E(j_lab, np.array(ppg_data[color_name]['LAB']))
            print('JPEG dE: ', dE_j)

        # vis_s = np.tile(c_rgb, [300, 300, 1])
        # vis_d = np.tile(d_rgb, [300, 300, 1])
        # vis_ppg = np.tile(ppg_data[color_name]['RGB'], [300, 300, 1])
        # if path_jpeg:
        #     vis_j = np.tile(j_rgb, [300, 300, 1])
        # if path_jpeg:
        #     fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
        #     ax1.imshow(vis_s)
        #     ax1.set_title('Custom wb')
        #     ax2.imshow(vis_ppg)
        #     ax2.set_title('Gt')
        #     ax3.imshow(vis_d)
        #     ax3.set_title('Default wb')
        #     ax4.imshow(vis_j)
        #     ax4.set_title('jpeg')
        #     plt.show()
        return dE_d, dE_c, dE_j, d_rgb, c_rgb, j_rgb, d_xyz, c_xyz

def deltaE(lab1, lab2):
    return np.sum((np.array(lab1) - np.array(lab2)) ** 2) ** 0.5

def read_ppg_data():
    return ref2xyz.read_data('VOC 2014 Color Data.csv')

def average_patch(patch):
    mr = mg = mb = 0.0
    for i in range(0, np.shape(patch)[0]):
        for j in range(0, np.shape(patch)[1]):
            mr += patch[i][j][0]
            mg += patch[i][j][1]
            mb += patch[i][j][2]
    s = np.shape(patch)[0] * np.shape(patch)[1]
    return np.array([mr / s, mg / s, mb / s])

#read data
ppg_data = read_ppg_data()
path_im = 'E:/UIUC/Data_11_07_18/Android'
with open('E:/UIUC/Data_11_07_18/AndroidNoFlashLabel.json', 'r') as fp:
    labels = json.load(fp)

mdE_d = 0.0
mdE_c = 0.0
mdE_j = 0.0
dE_ds = []
dE_cs = []
dE_js = []
gp = open('ResultAndroid.txt', 'w')
gp.write('%d\n' % len(labels.keys()))
cnt = 0
print(len(labels.keys()))
for key in labels.keys():
    cnt += 1
    im_name = labels[key][0]['im_name']
    patch_roi = labels[key][0]['patch_roi']
    wb_roi = labels[key][0]['wb_roi']
    print(cnt, ' ', im_name)
    path_jpeg = path_im + '/Camera/' + 'JPEG' + im_name[3:-3] + 'jpg'
    dE_d, dE_c, dE_j, d_rgb, c_rgb, j_rgb, d_xyz, c_xyz = test_color(path_im + '/' + im_name, key, ppg_data, wb_roi, patch_roi, path_jpeg)
    # print(dE_d)
    # print(dE_c)
    # print(dE_j)
    gp.write('%s, %f, %f, %f, %d, %d, %d, %f, %f, %f, %f, %f, %f\n' % (key.replace(' ', '_'), dE_d, dE_c, dE_j, j_rgb[0], j_rgb[1], j_rgb[2], d_xyz[0], d_xyz[1], d_xyz[2], c_xyz[0], c_xyz[1], c_xyz[2]))
    mdE_d += dE_d
    mdE_c += dE_c
    mdE_j += dE_j
    dE_ds.append(dE_d)
    dE_cs.append(dE_c)
    dE_js.append(dE_j)

print(cnt - 1)
mdE_c /= len(labels.keys())
mdE_d /= len(labels.keys())
mdE_j /= len(labels.keys())
print('Mean delta E custom wb:  ', mdE_c)
print('Mean delta E default wb: ', mdE_d)
print('Mean delta E camera jpg: ', mdE_j)
gp.write('%f\n' % mdE_d)
gp.write('%f\n' % mdE_c)
gp.write('%f\n' % mdE_j)
gp.close()
plt.plot(range(0, len(dE_cs)), dE_ds, label = 'default wb')
plt.plot(range(0, len(dE_cs)), dE_cs, label = 'custom wb')
plt.plot(range(0, len(dE_cs)), dE_js, label = 'jpeg')
plt.legend(loc='upper left')
plt.show()
