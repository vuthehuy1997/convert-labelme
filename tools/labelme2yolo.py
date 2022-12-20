import os
import json
import cv2
import numpy as np
from PIL import Image
import base64
import io
import argparse

from sklearn.model_selection import train_test_split

from utils import string2rgb, create_folder

CUR_DIR = os.getcwd()


def points2yolo(points, shape_type, image_width, image_height):
    if shape_type == 'rectangle':
        top_left = points[0]
        bottom_right = points[1]
    else:   
        points = np.array(points)
        top_left = [min(points[:, 0]), min(points[:, 1])]
        bottom_right = [max(points[:, 0]), max(points[:, 1])]
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]
    center = (int(top_left[0] + width / 2), int(top_left[1] + height / 2))
    x = center[0] / image_width
    y = center[1] / image_height
    w = width / image_width
    h = height / image_height
    return x, y, w, h


def convertlabelme2yolo(label_filepath, txt_filepath, subset):
    txt_file = open(txt_filepath, 'w')
    data = json.load(open(label_filepath, 'r'))
    image_data = data['imageData']
    image_height = data['imageHeight']
    image_width = data['imageWidth']
    image_path = data['imagePath']
    if image_data != None:
        img = string2rgb(image_data)
    else:
        print(os.path.join(LABELME_DIR, image_path))
        img = cv2.imread(os.path.join(LABELME_DIR, image_path))
    print(img.shape)
    image_filename = data['imagePath']
    cv2.imwrite(os.path.join(YOLO_DIR, 'images', subset, image_filename), img)
    shapes = data['shapes']
    for shape in shapes:
        label = shape['label']
        shape_type = shape['shape_type']
        points = shape['points']
        print(shape_type, points)
        x, y, w, h = points2yolo(
            points, shape_type, image_width, image_height
        )
        print(x, y, w, h, label)
        txt_file.write(f'{label} {x} {y} {w} {h}\n')
    txt_file.close()


if __name__ == '__main__':
    args = argparse.ArgumentParser(
        description='Create dataset Text Line Detection with YOLO format for Cavet OCR'
    )
    args.add_argument(
        '-lbme_dir', '--labelme_dir', 
        default='labelme', type=str,
        help='Path to LabelMe directory'
    )
    args.add_argument(
        '-yolo_dir', '--yolo_dir', 
        default='yolo', type=str,
        help='Path to YOLO directory'
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
    LABELME_DIR = os.path.join(CUR_DIR, args.labelme_dir)
    YOLO_DIR = os.path.join(CUR_DIR, args.yolo_dir)
    test_size = float(args.test_size)
    random_state = int(args.random_state)
    create_folder([YOLO_DIR])
    is_split = False
    if args.split is True:
        is_split = True
    label_list = os.listdir(LABELME_DIR)
    label_list = list(filter(lambda label: label.endswith('json'), label_list))
    if is_split is True:
        train_list, test_list = train_test_split(
            label_list, test_size=test_size, random_state=random_state
        )
        need_create_folders = []
        for subset in ['train', 'test']:
            need_create_folders.append(os.path.join(YOLO_DIR, 'images', subset))
            need_create_folders.append(os.path.join(YOLO_DIR, 'labels', subset))
        create_folder(need_create_folders)
        for label_filename in train_list:
            txt_filename = label_filename.replace('json', 'txt')
            label_filepath = os.path.join(LABELME_DIR, label_filename)
            txt_filepath = os.path.join(YOLO_DIR, 'labels', 'train', txt_filename)
            convertlabelme2yolo(label_filepath, txt_filepath, 'train') 
            print('train', label_filepath, txt_filepath)
        for label_filename in test_list:
            txt_filename = label_filename.replace('json', 'txt')
            label_filepath = os.path.join(LABELME_DIR, label_filename)
            txt_filepath = os.path.join(YOLO_DIR, 'labels', 'test', txt_filename)
            convertlabelme2yolo(label_filepath, txt_filepath, 'test') 
            print('test', label_filepath, txt_filepath)
    else:
        TOTAL_IMAGES = os.path.join(YOLO_DIR, 'images', 'total')
        TOTAL_LABELS = os.path.join(YOLO_DIR, 'labels', 'total')
        create_folder([TOTAL_IMAGES, TOTAL_LABELS])
        for label_filename in label_list:
            txt_filename = label_filename.replace('json', 'txt')
            label_filepath = os.path.join(LABELME_DIR, label_filename)
            txt_filepath = os.path.join(YOLO_DIR, 'labels', 'total', txt_filename)
            convertlabelme2yolo(label_filepath, txt_filepath, 'total') 
            print('total', label_filepath, txt_filepath)