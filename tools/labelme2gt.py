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

def labelme2gt(label_filepath, subset):
    data = json.load(open(label_filepath, 'r'))
    image_data = data['imageData']
    img = string2rgb(image_data)
    image_path = data['imagePath']
    image_filename = image_path[:-4]
    label_filename = image_filename + '.json'
    cv2.imwrite(os.path.join(GT_DIR, subset, image_path), img)
    shapes = data['shapes']
    gt_dict = {}
    for idx, shape in enumerate(shapes):    
        label = shape['label']
        print(label)
        if '###' in label:
            continue
        if len(label.split('_')) == 3:
            label_text, name, is_value = label.split('_')
        else:
            label_text, name = label.split('_')
            is_value = '0'
        if is_value == '1':
            gt_dict[name] = label_text
    with open(os.path.join(GT_DIR, subset, label_filename), 'w') as f:
        json.dump(gt_dict, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    args = argparse.ArgumentParser(
        description='Create dataset end2end for Cavet OCR'
    )
    args.add_argument(
        '-lbme_dir', '--labelme_dir', 
        default='labelme', type=str,
        help='Path to LabelMe directory'
    )
    args.add_argument(
        '-gt_dir', '--groundtruth_dir', 
        default='gt', type=str,
        help='Path to Groundtruth directory'
    )
    args.add_argument(
        '-meta', '--meta_filename', 
        default='meta.txt', type=str,
        help='meta.txt file path'
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
    GT_DIR = os.path.join(CUR_DIR, args.groundtruth_dir)
    TRAIN_GT_DIR = os.path.join(GT_DIR, 'train')
    TEST_GT_DIR = os.path.join(GT_DIR, 'test')
    ALL_GT_DIR = os.path.join(GT_DIR, 'all')
    meta_filename = os.path.join(CUR_DIR, args.meta_filename)
    test_size = float(args.test_size)
    random_state = int(args.random_state)
    if args.split is True:
        is_split = True
        need_create_folders = [GT_DIR, TRAIN_GT_DIR, TEST_GT_DIR]
    else:
        is_split = False
        need_create_folders = [GT_DIR, ALL_GT_DIR]
    create_folder(need_create_folders)
    meta_lines = open(meta_filename, 'r').readlines()
    meta = []
    for idx, meta_line in enumerate(meta_lines):
        value = meta_line[:-1]
        meta.append({'id': idx, 'name': value})
    with open(os.path.join(GT_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)
    label_list = os.listdir(LABELME_DIR)
    label_list = list(filter(lambda label: label.endswith('json'), label_list))
    if is_split is True:
        train_list, test_list = train_test_split(
            label_list, test_size=test_size, random_state=random_state
        )
        for label_filename in train_list:
            label_filepath = os.path.join(LABELME_DIR, label_filename)
            print(label_filepath)
            labelme2gt(label_filepath, 'train')
        for label_filename in test_list:
            label_filepath = os.path.join(LABELME_DIR, label_filename)
            print(label_filepath)
            labelme2gt(label_filepath, 'test')
    else:
        for label_filename in label_list:
            label_filepath = os.path.join(LABELME_DIR, label_filename)
            print(label_filepath)
            labelme2gt(label_filepath, 'all')
