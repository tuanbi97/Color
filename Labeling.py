import io
import rawpy
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from google.cloud import vision
from google.cloud.vision import types
from google.oauth2 import service_account
import json
import glob


def GoogleOCR(rgb):
    #rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    #rgb = cv2.equalizeHist(rgb)
    cv2.imwrite('tmp.png', rgb)
    with io.open('tmp.png', 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content = content)
    image_context = vision.types.ImageContext(language_hints=['en-t-i0-handwrit'])

    response = client.document_text_detection(image=image, image_context=image_context)

    #print(response.full_text_annotation)
    return response

def solve_response(res, id):
    p = res.text_annotations
    if not p:
        return 'NA', []
    text = p[0].description
    #print(p[0])
    stext = text.split('\n')
    if len(stext) != 3:
        return 'NA', []
    stext = stext[0].split('-')
    if len(stext) != 2 or stext[0] == '' or stext[1] == '':
        return 'NA', []
    #print(p)
    minx = miny = 1000
    maxx = maxy = 0
    for i in range(0, len(p[0].bounding_poly.vertices)):
        vertex = p[0].bounding_poly.vertices[i]
        minx = min(minx, vertex.x)
        miny = min(miny, vertex.y)
        maxx = max(maxx, vertex.x)
        maxy = max(maxy, vertex.y)
    return text, [crop_ocr[id][1] + minx, crop_ocr[id][0] + miny, crop_ocr[id][1] + maxx, crop_ocr[id][0] + maxy]

crop_ocr = [
    [2088, 1750, 2492, 2206]
]

roi = [
    [1900, 1650, 2200, 1950]
]
wb_roi = [
    [1950, 1160, 2150, 1360]
]

credentials = service_account.Credentials.from_service_account_file('E:/UIUC/My Project 21860-8bd26a835a0d.json')
client = vision.ImageAnnotatorClient(credentials=credentials)

jsonfile = 'IOSNoFlashLabel.json'
datadir = 'E:/UIUC/Data_11_07_18/iOS'
im_names = glob.glob(datadir + '/*noflash.dng')
print(len(im_names))
#im_names = os.listdir(datadir)
sorted(im_names)
labels = {}
for pp in range(0, len(im_names)):
    imdir = im_names[pp]
    im_name = imdir[len(datadir) + 1:]
    print(im_name)
    #labels[im_name] = []
    raw = rawpy.imread(imdir)
    trgb = raw.postprocess(use_camera_wb=True)
    # plt.imshow(trgb)
    # plt.show()

    for i in range(0, 1):
        crop_region = crop_ocr[i]
        print(crop_region)
        rgb = trgb[crop_region[1]: crop_region[3], crop_region[0] : crop_region[2]]
        # plt.imshow(trgb[roi[0][1]:roi[0][3], roi[0][0]:roi[0][2]])
        # plt.imshow(rgb)
        # plt.show()
        response = GoogleOCR(rgb)
        label, ocr_box = solve_response(response, i)
        print(label)
        if label != 'NA':
            labels[label.split('\n')[1].upper()] = [{
                'im_name': im_name, 'patch_roi': roi[0], 'wb_roi': wb_roi[0]
            }]
        else:
            if 'NA' in labels.keys():
                labels['NA'].append({'im_name': im_name, 'patch_roi': roi[0], 'wb_roi': wb_roi[0]})
            else:
                labels['NA'] = [{'im_name': im_name, 'patch_roi': roi[0], 'wb_roi': wb_roi[0]}]



        #process_roi(trgb)
    # trgb = cv2.resize(trgb, (int(np.shape(trgb)[1]/2), int(np.shape(trgb)[0]/2)))
    # fig_size = plt.rcParams["figure.figsize"]
    # fig_size[0] = 6
    # fig_size[1] = 6
    # plt.rcParams["figure.figsize"] = fig_size
    # plt.imshow(trgb)
    # plt.show()
    raw.close()
with open(jsonfile, 'w') as fp:
    json.dump(labels, fp, indent=2)