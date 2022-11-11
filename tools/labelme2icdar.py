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


def convertlabelme2icdar(filepath, txt_filename):
    f = open(filepath, "r")
    data = json.load(f)
    txt_file = open(txt_filename, 'w')
    image_filename = data["imagePath"]
    image_data = data["imageData"]
    img = string2rgb(image_data)
    cv2.imwrite(os.path.join(IMAGES_DIR, image_filename), img)
    shapes = data["shapes"]
    for shape in shapes:
        shape_type = shape["shape_type"]
        points = shape["points"]
        label = shape["label"]
        label_text = label.split("_")[0]
        # if '###' in label_text:
        #     continue
        p0 = None
        p1 = None
        p2 = None
        p3 = None
        if shape_type == "polygon":
            p0 = (int(points[0][0]), int(points[0][1]))
            p1 = (int(points[1][0]), int(points[1][1]))
            p2 = (int(points[2][0]), int(points[2][1]))
            p3 = (int(points[3][0]), int(points[3][1]))
        elif shape_type == "rectangle":
            x0 = int(points[0][0])
            y0 = int(points[0][1])
            x1 = int(points[1][0])
            y1 = int(points[1][1])
            p0 = (x0, y0)
            p1 = (x1, y0)
            p2 = (x1, y1)
            p3 = (x0, y1)
        points = [
            [p0[0], p0[1]],
            [p1[0], p1[1]],
            [p2[0], p2[1]],
            [p3[0], p3[1]]
        ]
        txt_file.write(f"{p0[0]},{p0[1]},{p1[0]},{p1[1]},{p2[0]},{p2[1]},{p3[0]},{p3[1]},{label_text}\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description='Create dataset ICDAR format for Cavet OCR'
    )
    args.add_argument(
        '-lbme_dir', '--labelme_dir', 
        default='labelme', type=str,
        help='Path to LabelMe directory'
    )
    args.add_argument(
        '-icdar_dir', '--icdar_dir', 
        default='cavet_icdar', type=str,
        help='Path to Groundtruth directory'
    )
    args.add_argument(
        '-split', action='store_true',
        help='Is splitting dataset or not'
    )
    args.add_argument(
        '-ts', '--test_size', 
        default=0.2, type=str,
        help='Test size for splitting dataset'
    )
    args.add_argument(
        '-rs', '--random_state', 
        default=42, type=str,
        help='Random state'
    )
    args = args.parse_args()
    print(args)
    LABELME_DIR = os.path.join(CUR_DIR, args.labelme_dir)
    ICDAR_DIR = os.path.join(CUR_DIR, args.icdar_dir)
    test_size = float(args.test_size)
    random_state = int(args.random_state)
    IMAGES_DIR = os.path.join(ICDAR_DIR, 'images')
    is_split = False
    if args.split is True:
        is_split = True
    create_folder([ICDAR_DIR, IMAGES_DIR])

    label_list = os.listdir(LABELME_DIR)
    label_list = list(filter(lambda label: label.endswith('json'), label_list))
    if is_split is True:
        create_folder([ICDAR_DIR, IMAGES_DIR])
        TRAIN_ICDAR_DIR = os.path.join(ICDAR_DIR, 'train')
        TEST_ICDAR_DIR = os.path.join(ICDAR_DIR, 'test')
        create_folder([TRAIN_ICDAR_DIR, TEST_ICDAR_DIR])
        train_list, test_list = train_test_split(
            label_list, test_size=test_size, random_state=random_state
        )
        for filename in train_list:
            print('train', filename)
            txt_filename = filename[:-4] + 'txt'
            convertlabelme2icdar(
                os.path.join(LABELME_DIR, filename),
                os.path.join(TRAIN_ICDAR_DIR, txt_filename)
            )
        for filename in test_list:
            print('test', filename)
            txt_filename = filename[:-4] + 'txt'
            convertlabelme2icdar(
                os.path.join(LABELME_DIR, filename),
                os.path.join(TEST_ICDAR_DIR, txt_filename)
            )
    else:
        TOTAL_ICDAR_DIR = os.path.join(ICDAR_DIR, 'total')
        create_folder([TOTAL_ICDAR_DIR])
        for filename in label_list:
            print('total', filename)
            txt_filename = filename[:-4] + 'txt'
            convertlabelme2icdar(
                os.path.join(LABELME_DIR, filename),
                os.path.join(TOTAL_ICDAR_DIR, txt_filename)
            )
