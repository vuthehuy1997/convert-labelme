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

def labelme2recog(label_filepath):#, wrong_text, new_label):
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
        if str(is_value) == '0':
            is_value = 'k'
        if str(is_value) == '1':
            is_value = 'v'
        if str(name) == '33':
            is_value = 'k'
        label_text = label_text.replace('\n', '')
        shapes[idx]['label'] = '_'.join([label_text, name, is_value])
        # if label_text == wrong_text:
        #     shapes[idx]['label'] = '_'.join([label_text, name, new_label])
    with open(label_filepath, 'w') as f:
        json.dump(data, f)

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
