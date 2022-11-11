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

def convertlabelme2pick(filepath, i, txt_file=None):
    filename = filepath.split('/')[-1]
    name = 'img' + '%04d' % i
    print('basename', name)
    f = open(filepath, "r")
    data = json.load(f)
    image_data = data["imageData"]
    img = string2rgb(image_data)
    cv2.imwrite(os.path.join(IMAGES_DIR, name + '.jpg'), img)
    shapes = data["shapes"]
    tsv_file = open(os.path.join(BOXES_AND_TRANS_DIR, name + '.tsv'), 'w')
    entities = {}
    for idx, shape in enumerate(shapes):
        shape_type = shape["shape_type"]
        points = shape["points"]
        label = shape["label"]
        print(label)
        if label == '###':
            continue
        label_text = label.split("_")[0]
        # if '###' in label_text:
        #     continue
        if len(label.split('_')) == 3:
            label_text, column_name, is_value = label.split('_')
        else:
            label_text, column_name = label.split('_')
            is_value = '0'
        column_name = int(column_name)
        transcripts = label_text
        box_entity_types = 'other' if (is_value == '0' or column_name == 26) else column_name
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
        points = [[p0[0], p0[1]], [p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]]]
        if box_entity_types != 'other':
            entities[column_name] = transcripts
        tsv_file.write(
            f'{idx},{p0[0]},{p0[1]},{p1[0]},{p1[1]},{p2[0]},{p2[1]},{p3[0]},{p3[1]},{transcripts},{box_entity_types}\n'
        ) 
    tsv_file.close()
    with open(os.path.join(ENTITIES_DIR, name + '.txt'), 'w', encoding='utf8') as json_file:
        json.dump(entities, json_file, indent=4, ensure_ascii=False)
    line = f"{i},cavet,{name}\n"
    LINES.append(line)
    if txt_file:
        txt_file.write(line)


def create_test(lines):
    for line in lines:
        line = line[:-1]
        i, document_type, name = line.split(',')
        tsv_file = open(os.path.join(BOXES_AND_TRANS_DIR, name + '.tsv'), 'r')
        img = cv2.imread(os.path.join(IMAGES_DIR, name + '.jpg'))
        cv2.imwrite(os.path.join(TEST_IMAGES_DIR, name + '.jpg'), img)
        boxes_and_trans_lines = tsv_file.readlines()
        test_tsv_file = open(
            os.path.join(TEST_BOXES_AND_TRANS_DIR, name + '.tsv'), 'w'
        )
        for boxes_and_trans_line in boxes_and_trans_lines:
            if boxes_and_trans_line[-1] == '\n':
                boxes_and_trans_line = boxes_and_trans_line[:-1]
            last_comma = boxes_and_trans_line.rfind(',')
            boxes_and_trans_drop_column_name = boxes_and_trans_line[:last_comma]
            # idx, *points, transcript, column_name = boxes_and_trans_line.split(',')
            test_tsv_file.write(boxes_and_trans_drop_column_name + '\n')
        test_tsv_file.close()

def create_dataset():
    need_create_folders = [
        PICK_DIR, IMAGES_DIR, BOXES_AND_TRANS_DIR, ENTITIES_DIR
    ]
    create_folder(need_create_folders)
    label_list = os.listdir(LABELME_DIR)
    label_list = list(filter(lambda label: label.endswith('json'), label_list))
    if is_split is True:
        train_list, test_list = train_test_split(
            label_list, test_size=test_size, random_state=random_state
        )
        TRAIN_FILE = io.open(
            os.path.join(
                PICK_DIR, "train_samples_list.csv"
            ),
            "w", encoding="utf8"
        )
        TEST_FILE = io.open(
            os.path.join(
                PICK_DIR, "test_samples_list.csv"
            ),
            "w", encoding="utf8"
        )
        idx = 0

        for filename in train_list:
            print('train', filename)
            convertlabelme2pick(
                os.path.join(LABELME_DIR, filename), idx, TRAIN_FILE
            )
            idx += 1
        for filename in test_list:
            print('test', filename)
            convertlabelme2pick(
                os.path.join(LABELME_DIR, filename), idx, TEST_FILE
            )
            idx += 1
        TRAIN_FILE.close()
        TEST_FILE.close()
    else:
        TOTAL_FILE = io.open(
            os.path.join(
                PICK_DIR, "total_samples_list.csv"
            ),
            "w", encoding="utf8"
        )
        for idx, filename in enumerate(label_list):
            print('all', filename)
            convertlabelme2pick(
                os.path.join(LABELME_DIR, filename), idx, TOTAL_FILE
            )
        TOTAL_FILE.close()


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
        '-pick_dir', '--pick_dir',
        default='cavet_pick', type=str,
        help='Path to PICK directory'
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
    args.add_argument(
        '-create_infer', '--create_inference',
        action='store_true',
        help='Is creating a inference folder or not'
    )
    args = args.parse_args()
    LABELME_DIR = os.path.join(CUR_DIR, args.labelme_dir)
    PICK_DIR = os.path.join(CUR_DIR, args.pick_dir)
    test_size = float(args.test_size)
    random_state = int(args.random_state)
    IMAGES_DIR = os.path.join(PICK_DIR, 'images')
    BOXES_AND_TRANS_DIR = os.path.join(PICK_DIR, 'boxes_and_transcripts')
    ENTITIES_DIR = os.path.join(PICK_DIR, 'entities')
    is_split = False
    is_create_infer = False
    if args.split is True:
        is_split = True
    if args.create_inference is True:
        is_create_infer = True

    if is_create_infer is False:
        print('Create PICK dataset')
        create_dataset()
    else:
        print('Create test folder')
        TEST_DIR = os.path.join(CUR_DIR, args.pick_dir + '_infer')
        TEST_IMAGES_DIR = os.path.join(TEST_DIR, 'images')
        TEST_BOXES_AND_TRANS_DIR = os.path.join(
            TEST_DIR, 'boxes_and_transcripts'
        )
        need_create_folders = [
            TEST_DIR, TEST_IMAGES_DIR, TEST_BOXES_AND_TRANS_DIR,
        ]
        create_folder(need_create_folders)
        create_dataset()
        create_test(LINES)
        shutil.rmtree(PICK_DIR)
