import os
import json
import cv2
import numpy as np
from PIL import Image
import base64
import io
import random
from sklearn.model_selection import train_test_split
import argparse

from utils import string2rgb, create_folder

CUR_DIR = os.getcwd()

LINES = []
Y = []

def string2rgb(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

def four_point_transform(image, pts):
    (tl, tr, br, bl) = pts

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(np.float32(pts), dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def labelme2recog(label_filepath):
    data = json.load(open(label_filepath, 'r'))
    image_data = data['imageData']
    # image_height = data['imageHeight']
    # image_width = data['imageWidth']
    img = string2rgb(image_data)
    image_path = data['imagePath']
    image_filename = image_path[:-4]
    shapes = data['shapes']
    for idx, shape in enumerate(shapes):    
        shape_type = shape['shape_type']
        points = shape['points']
        label = shape['label']
        label_text = label.split('_')[0]
        print(label)
        if '###' in label_text:
            continue
        if len(label.split('_')) == 3:
            label_text, name, is_value = label.split('_')
        else:
            label_text, name = label.split('_')
            is_value = '0'
        if is_value not in ['0', '1']:
            print('check', is_value)
        Y.append(int(is_value))
        p0 = None
        p1 = None 
        p2 = None 
        p3 = None
        if shape_type == 'polygon':
            p0 = (int(points[0][0]), int(points[0][1]))
            p1 = (int(points[1][0]), int(points[1][1]))
            p2 = (int(points[2][0]), int(points[2][1]))
            p3 = (int(points[3][0]), int(points[3][1]))
        elif shape_type == 'rectangle':
            x0 = int(points[0][0])
            y0 = int(points[0][1])
            x1 = int(points[1][0])
            y1 = int(points[1][1])
            p0 = (x0, y0)
            p1 = (x1, y0)
            p2 = (x1, y1)
            p3 = (x0, y1)
        crop = four_point_transform(img, (p0, p1, p2, p3))
        image_crop_filename = f'{image_filename}_crop_{idx}.jpg'
        cv2.imwrite(os.path.join(RECOG_IMAGES_DIR, image_crop_filename), crop)
        crop_filepath = os.path.join('images', image_crop_filename)
        line = f'{crop_filepath}\t{label_text}\n'
        LINES.append(line)
        RECOG_GT_FILE.write(line)


if __name__ == '__main__':
    args = argparse.ArgumentParser(
        description='Create dataset Recognition for Cavet OCR'
    )
    args.add_argument(
        '-lbme_dir', '--labelme_dir', 
        default='labelme', type=str,
        help='Path to LabelMe directory'
    )
    args.add_argument(
        '-recog_dir', '--recognition_dir', 
        default='recog', type=str,
        help='Path to Recognition directory'
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
    RECOG_DIR = os.path.join(CUR_DIR, args.recognition_dir)
    RECOG_IMAGES_DIR = os.path.join(RECOG_DIR, 'images')
    create_folder([RECOG_DIR, RECOG_IMAGES_DIR])
    test_size = float(args.test_size)
    random_state = int(args.random_state)
    is_split = False
    if args.split is True:
        is_split = True
    RECOG_GT_FILE = open(os.path.join(RECOG_DIR, 'gt.txt'), 'w')
    TRAIN_GT_FILE = open(os.path.join(RECOG_DIR, 'train.txt'), 'w')
    TEST_GT_FILE = open(os.path.join(RECOG_DIR, 'test.txt'), 'w')

    label_list = os.listdir(LABELME_DIR)
    label_list = list(filter(lambda label: label.endswith('json'), label_list))
    for label_filename in label_list:
        label_filepath = os.path.join(LABELME_DIR, label_filename)
        print(label_filepath)
        labelme2recog(label_filepath)
    RECOG_GT_FILE.close()
    if is_split is True:
        train_recog, test_recog = train_test_split(
            LINES, test_size=test_size, random_state=random_state, stratify=Y
        )
        for line in train_recog:
            TRAIN_GT_FILE.write(line)
        for line in test_recog:
            TEST_GT_FILE.write(line)
        TRAIN_GT_FILE.close()
        TEST_GT_FILE.close()
