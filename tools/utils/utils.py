import os
import json
import cv2
import numpy as np
from PIL import Image
import base64
import io
import argparse

def string2rgb(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

def create_folder(folder_list):
    for folder_path in folder_list:
        if os.path.exists(folder_path) is False:
            os.makedirs(folder_path)
