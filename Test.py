from PIL import Image
import exiftool
import numpy as np

path = '../RAWImages/RAW_2018_08_31_10_25_09_827.dng'
with exiftool.ExifTool() as et:
    metadata = et.get_metadata(path)
    tmp = np.array(metadata.keys())
    tmp.sort()
    print(len(tmp))
    for key in tmp:
        print(key)

