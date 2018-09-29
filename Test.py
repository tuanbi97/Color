import codecs
import sys

from PIL import Image
import exiftool
import numpy as np
import json

def _fscodec():
    encoding = sys.getfilesystemencoding()
    errors = "strict"
    if encoding != "mbcs":
        try:
            codecs.lookup_error("surrogateescape")
        except LookupError:
            pass
        else:
            errors = "surrogateescape"

    def fsencode(filename):
        """
        Encode filename to the filesystem encoding with 'surrogateescape' error
        handler, return bytes unchanged. On Windows, use 'strict' error handler if
        the file system encoding is 'mbcs' (which is the default encoding).
        """
        if isinstance(filename, bytes):
            return filename
        else:
            return filename.encode(encoding, errors)

    return fsencode

fsencode = _fscodec()
del _fscodec

path = '../RAWImages/RAW_2018_08_31_10_25_09_827.dng'
with exiftool.ExifTool() as et:
    #tmp = json.loads(et.execute(b"-j", path).decode("utf-8"))[0]
    #print(tmp)
    tmp = et.execute(b"-b -EXIF:OpcodeList2", path)
    print(tmp)

