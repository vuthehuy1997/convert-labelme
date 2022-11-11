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


def convertlabelme2icdar(filepath, txt_file):
    f = open(filepath, "r")
    data = json.load(f)
    image_filename = data["imagePath"]
    image_data = data["imageData"]
    img = string2rgb(image_data)
    cv2.imwrite(os.path.join(IMAGES_DIR, image_filename), img)
    shapes = data["shapes"]
    json_data = {
        "file_name": os.path.join('image_files', image_filename),
        "height": img.shape[0],
        "width": img.shape[1],
    }
    annotations = []
    for shape in shapes:
        shape_type = shape["shape_type"]
        points = shape["points"]
        label = shape["label"]
        if label == '###':
            continue
        label_text = label.split("_")[0]
        if len(label.split('_')) == 3:
            label_text, column_name, is_value = label.split('_')
        else:
            label_text, column_name = label.split('_')
            is_value = '0'
        column_name = int(column_name)
        label_text = label_text.replace('###', '')
        box_entity_type = 26 if (is_value == '0' or column_name == 26) else column_name
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
            p0[0], p0[1],
            p1[0], p1[1],
            p2[0], p2[1],
            p3[0], p3[1]
        ]
        points = list(map(lambda x: float(x), points))
        annotation = {
            "text": label_text, "box": points, "label": box_entity_type
        }
        annotations.append(annotation)
    json_data["annotations"] = annotations
    json_string = json.dumps(json_data, ensure_ascii=False).encode('utf8')
    json_string = json_string.decode()
    line = f"{json_string}\n"
    txt_file.write(line)


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description='Create dataset KIE MMLAB format for Cavet OCR'
    )
    args.add_argument(
        '-lbme_dir', '--labelme_dir', 
        default='labelme', type=str,
        help='Path to LabelMe directory'
    )
    args.add_argument(
        '-mm_dir', '--mm_dir', 
        default='mm', type=str,
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
    MM_DIR = os.path.join(CUR_DIR, args.mm_dir)
    test_size = float(args.test_size)
    random_state = int(args.random_state)
    meta_filename = os.path.join(CUR_DIR, args.meta_filename)
    IMAGES_DIR = os.path.join(MM_DIR, 'image_files')
    is_split = False
    if args.split is True:
        is_split = True
    create_folder([MM_DIR, IMAGES_DIR])

    label_list = os.listdir(LABELME_DIR)
    label_list = list(filter(lambda label: label.endswith('json'), label_list))
    if is_split is True:
        train_list, test_list = train_test_split(
            label_list, test_size=test_size, random_state=random_state
        )
        TRAIN_FILE = io.open(
            os.path.join(MM_DIR, "train.txt"), "w", encoding="utf8"
        )
        TEST_FILE = io.open(
            os.path.join(MM_DIR, "test.txt"), "w", encoding="utf8"
        )
        for filename in train_list:
            print('train', filename)
            convertlabelme2icdar(
                os.path.join(LABELME_DIR, filename), TRAIN_FILE
            )
        for filename in test_list:
            print('test', filename)
            convertlabelme2icdar(
                os.path.join(LABELME_DIR, filename), TEST_FILE
            )
        TRAIN_FILE.close()
        TEST_FILE.close()
    else:
        ALL_FILE = io.open(
            os.path.join(MM_DIR, "gt.txt"), "w", encoding="utf8"
        )
        for filename in label_list:
            print('all', filename)
            convertlabelme2icdar(os.path.join(LABELME_DIR, filename), ALL_FILE)
        ALL_FILE.close()
    meta_lines = open(meta_filename, 'r').readlines()
    class_list_file = open(os.path.join(MM_DIR, 'class_list.txt'), 'w')
    for idx, meta_line in enumerate(meta_lines):
        class_list_file.write(f'{idx} {meta_line}')
    class_list_file.close()
    keys = open('keys.txt', 'r').readlines()
    keys = keys[0]
    keys = keys[:-1]
    dict_file = open(os.path.join(MM_DIR, 'dict.txt'), 'w')
    for key in keys:
        dict_file.write(key + '\n')
    dict_file.close()
