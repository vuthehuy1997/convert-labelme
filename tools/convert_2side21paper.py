import cv2
from PIL import Image
import json
import argparse
import os
import numpy as np
import shutil

def read2replace_labelme(label_filepath):
    rs = {}
    data = json.load(open(label_filepath, 'r'))
    list_object = data['shapes']
    for oj in list_object:
        if oj['shape_type'] == 'polygon':
            bbox = oj['points']
        elif oj['shape_type'] == 'rectangle':
            tl = oj['points'][0]
            br = oj['points'][1]
            bbox = [tl, [br[0], tl[1]], br, [tl[0], br[1]]]
        if oj['label'] == '0':
            rs['left'] = bbox  
        elif oj['label'] == '1':
            rs['right'] = bbox 
    data['shapes'] = [{
      "label": "0",
      "points": [rs['left'][0], rs['right'][1], rs['right'][2], rs['left'][3]],
      "group_id": None,
      "shape_type": "polygon",
      "flags": {}
    }]
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source', required=True,
        help='Folder dir', default='data/source/tmp'
    )
    parser.add_argument(
        '--dest', required=True,
        help='Folder dir', default='data/dest/tmp'
    )

    args = parser.parse_args()

    if os.path.exists(args.dest) is False:
        os.makedirs(args.dest, exist_ok=True)

    print(args.source)
    filenames = [f for f in os.listdir(args.source) if os.path.splitext(f)[1] == '.json']
    
    
    for idx, filename in enumerate(filenames):
        print(filename)
        file_path = os.path.join(args.source, filename)
        print(idx, file_path)
        
        json_data = read2replace_labelme(file_path)
        
        new_label_path = os.path.join(args.dest, filename)
        print(new_label_path)
        with open(new_label_path, 'w') as f_json:
            json.dump(json_data, f_json)
        
        
