import os
import json
import numpy as np
import base64
import io
import random
import argparse

from utils import create_folder

CUR_DIR = os.getcwd()

def labelme2recog(label_filepath, wrong_text, new_text='', new_class='', new_label=''):
    changed = False
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
        
        # if wrong_text in label_text:
        if wrong_text == label_text:
            if new_text == '':
                new_text = label_text
            if new_class == '':
                new_class = name
            if new_label == '':
                new_label = is_value
            shapes[idx]['label'] = '_'.join([new_text, new_class, new_label])
            changed = True
    with open(label_filepath, 'w') as f:
        json.dump(data, f)
    return changed

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

    args.add_argument(
        '-w', '--wrong_text', 
        default='', type=str,
        help='Path to Recognition directory'
    )
    args.add_argument(
        '-text', '--new_text', 
        default='', type=str,
        help='Path to Recognition directory'
    )
    args.add_argument(
        '-class', '--new_class', 
        default='', type=str,
        help='Path to Recognition directory'
    )
    args.add_argument(
        '-label', '--new_label', 
        default='', type=str,
        help='Path to Recognition directory'
    )
    args = args.parse_args()
    print(args)
    wrong_text = args.wrong_text
    new_text = args.new_text
    new_class = args.new_class
    new_label = args.new_label
    
    LABELME_DIR = os.path.join(CUR_DIR, args.labelme_dir)
    RECOG_DIR = os.path.join(CUR_DIR, args.recognition_dir)
    create_folder([RECOG_DIR])

    label_list = os.listdir(LABELME_DIR)
    label_list = list(filter(lambda label: label.endswith('json'), label_list))
    label_list.sort()

    count = 0
    for label_filename in label_list:
        label_filepath = os.path.join(LABELME_DIR, label_filename)
        print(label_filepath)
        changed = labelme2recog(label_filepath,wrong_text,new_text, new_class, new_label)
        if changed:
            count += 1
    print(count)