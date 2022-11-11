import shutil
import os
import json
import base64
import cv2
from PIL import Image
import io
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

from utils import string2rgb, create_folder

CUR_DIR = os.getcwd()

LINES = []


def convertlabelme2pick(filepath, category):
    filename = filepath.split('/')[-1]
    name = filename.split('.')[0]
    # print('basename', name)
    f = open(filepath, "r")
    data = json.load(f)
    image_data = data["imageData"]
    image_path = data["imagePath"]
    if image_data is not None:
        img = string2rgb(image_data)
    else:
        img = cv2.imread(os.path.join(LABELME_DIR, image_path))
    shapes = data["shapes"]
    entities = {}
    for idx, shape in enumerate(shapes):
        # shape_type = shape["shape_type"]
        # points = shape["points"]
        label = shape["label"]
        if '_' not in label:
            label_text = label
            box_entity_types = 'other'
        else:
            label_text = label.split("_")[0]
            box_entity_types = int(label.split("_")[1])
        transcripts = label_text
        column_name = box_entity_types
        if box_entity_types != 'other':
            entities[column_name] = transcripts
            if column_name == category:
                print(transcripts)
        # else:
        #     if '5. Người đại diện' in label_text:
        #         print(label_text)


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description='Create dataset end2end for Cavet OCR'
    )
    args.add_argument(
        '-lbme_dir', '--labelme_dir',
        default='labelme', type=str,
        help='Path to LabelMe directory'
    )
    args.add_argument(
        '-c', '--category',
        default='0', type=str,
        help='Category to check'
    )
    args = args.parse_args()
    LABELME_DIR = os.path.join(CUR_DIR, args.labelme_dir)
    category = int(args.category)
    label_list = os.listdir(LABELME_DIR)
    label_list = list(filter(lambda label: label.endswith('json'), label_list))
    for filename in label_list:
        convertlabelme2pick(os.path.join(LABELME_DIR, filename), category)
