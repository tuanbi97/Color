import codecs
import sys

from PIL import Image
import exiftool
import numpy as np
import json

path = 'E:/UIUC/Data_10_07_18/no_flash/RAW_2018_10_07_11_05_37_707.dng'
with exiftool.ExifTool() as et:
    tmp = et.get_metadata(path)
    for k in tmp.keys():
        print(k, ':', tmp[k])

