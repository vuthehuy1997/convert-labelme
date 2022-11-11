import os
import json
import numpy as np
import base64
import io
import random
import argparse

from utils import create_folder

CUR_DIR = os.getcwd()

LABELS = {}

def labelme2recog(label_filepath):
    print('--------',label_filepath)
    data = json.load(open(label_filepath, 'r'))
    shapes = data['shapes']
    for idx, shape in enumerate(shapes):    
        label = shape['label']
        label_text = label.split('_')[0]
        print(label)
        if '###' in label_text:
            continue
        if len(label.split('_')) == 3:
            label_text, name, is_value = label.split('_')
        else:
            label_text, name = label.split('_')
            is_value = 'unk'
        class_name = name + '_' + is_value
        label_text = os.path.basename(label_filepath) + '\t' + label_text
        if class_name not in LABELS:
            LABELS[class_name] = [label_text]
        else:
            LABELS[class_name].append(label_text)



if __name__ == '__main__':
    args = argparse.ArgumentParser(
        description='Create dataset Recognition Label for OCR'
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
    args = args.parse_args()
    print(args)
    LABELME_DIR = os.path.join(CUR_DIR, args.labelme_dir)
    RECOG_DIR = os.path.join(CUR_DIR, args.recognition_dir)
    create_folder([RECOG_DIR])

    label_list = os.listdir(LABELME_DIR)
    label_list = list(filter(lambda label: label.endswith('json'), label_list))
    label_list.sort()
    for label_filename in label_list:
        label_filepath = os.path.join(LABELME_DIR, label_filename)
        print(label_filepath)
        labelme2recog(label_filepath)

    for key in LABELS:
        with open(os.path.join(RECOG_DIR, 'trc_' + key + '.txt'), 'w') as f:
            for v in LABELS[key]:
                f.write(v + '\n')
