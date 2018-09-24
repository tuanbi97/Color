import rawpy
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pytesseract
from PIL import Image

print pytesseract.image_to_string(Image.open('test.png'))

imdir = '../RAWimages/RAW_2018_08_31_10_30_01_393.dng'

crop_ocr = [
    [900, 200, 400, 400],
    [900, 200 + 500, 400, 400],
    [900, 200 + 500*2, 400, 400],
    [900, 200 + 500*3, 400, 400],
    [900, 200 + 500*4, 400, 400],
    [900, 200 + 500*5, 400, 400],
    [900, 200 + 500*6, 400, 400],
]

raw = rawpy.imread(imdir)
trgb = raw.postprocess(use_camera_wb=True)
for i in range(0, 1):
    crop_region = crop_ocr[i]
    print(crop_region)
    rgb = trgb[crop_region[0]: crop_region[0] + crop_region[2], crop_region[1] : crop_region[1] + crop_region[3]]
    print(np.shape(rgb))
    print(pytesseract.image_to_string(Image.fromarray(rgb, 'RGB')))

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 9
    plt.rcParams["figure.figsize"] = fig_size
    plt.imshow(rgb)
    plt.show()
raw.close()
# plt.imshow(rgb)
# plt.show()

# p = raw.postprocess(output_bps=16, output_color = rawpy.ColorSpace.raw, no_auto_scale=True, no_auto_bright=True, gamma=(1,1))
# p = raw.postprocess(demosaic_algorithm=None,output_color = rawpy.ColorSpace.raw, output_bps=16)

# print(np.shape(p))
# print(np.max(p))