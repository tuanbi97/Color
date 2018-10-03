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

yt = 1330
xl = 800
step = 420
crop_ocr = [
    [yt, xl, 450, 300],
    [yt, xl + step, 450, 300],
    [yt, xl + step*2, 450, 300],
    [yt, xl + step*3, 450, 300],
    [yt, xl + step*4, 450, 300],
    [yt, xl + step*5, 450, 300],
    [yt, xl + step*6, 450, 300],
]

x, y = (1200, 450)
#
# roi = [
#     [x, y, 300, 300],
#     [x, y + 480, 300, 300],
#     [x, y + 480*2, 300, 300],
#     [x, y + 480*3, 300, 300],
#     [x, y + 480*4, 300, 300],
#     [x, y + 480*5, 300, 300],
#     [x, y + 480*6, 300, 300],
# ]

credentials = service_account.Credentials.from_service_account_file('E:/UIUC/My Project 21860-8bd26a835a0d.json')
client = vision.ImageAnnotatorClient(credentials=credentials)

jsonfile = 'IOSFlashLabel.json'
datadir = '../RawImages/iOS/flash/'
im_names = os.listdir(datadir)
sorted(im_names)
labels = {}
for pp in range(0, len(im_names) - 1):
    im_name = im_names[pp]
    print(im_name)
    #im_name = 'RAW_2018_08_31_10_25_18_676.dng'
    labels[im_name] = []
    imdir = datadir + im_name
    raw = rawpy.imread(imdir)
    trgb = raw.postprocess(use_camera_wb=True)

    for i in range(0, 7):
        crop_region = crop_ocr[i]
        print(crop_region)
        rgb = trgb[crop_region[0]: crop_region[0] + crop_region[2], crop_region[1] : crop_region[1] + crop_region[3]]
        # plt.imshow(rgb)
        # plt.show()
        #rgb = trgb[ROI[0]:ROI[0] + ROI[2], ROI[1] : ROI[1] + ROI[3]]
        #trgb = cv2.rectangle(trgb, (ROI[1], ROI[0]), (ROI[1] + ROI[3], ROI[0] + ROI[2]), (255, 0, 255), 3)
        response = GoogleOCR(rgb)
        label, ocr_box = solve_response(response, i)
        print(label)
        if label != 'NA':
            trgb = cv2.rectangle(trgb, (ocr_box[0], ocr_box[1]), (ocr_box[2], ocr_box[3]), (255, 0, 255), 3)
            print(ocr_box[3] - ocr_box[1])
            if (ocr_box[3] - ocr_box[1] < 250):
                roi = (ocr_box[0], ocr_box[3] + 5, ocr_box[0] + 300, ocr_box[3] + 5 + 300)
            else:
                roi = (ocr_box[2] + 5, ocr_box[1], ocr_box[2] + 5 + 300, ocr_box[1] + 300)
            #trgb = cv2.rectangle(trgb, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 255), 3)
            labels[im_name].append({'cname': label.split('\n')[0],
                                    'label' : label.split('\n')[1],
                                    'type': 'flash',
                                    'x1' : roi[0], 'y1' : roi[1], 'x2': roi[2], 'y2': roi[3]})
        else:
            labels[im_name].append({'cname': 'NA',
                                    'label': 'NA',
                                    'type': 'flash',
                                    'x1': 0, 'y1': 0, 'x2': 0,'y2': 0})

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