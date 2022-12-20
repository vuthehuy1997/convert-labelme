import cv2
from PIL import Image
import json
import argparse
import os
import numpy as np
import shutil

def four_point_transform(image, pts, bboxes):
    (tl, tr, br, bl) = pts
    
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([[0, 0], [max_width - 1 + 0, 0], [max_width - 1 + 0, max_height - 1 + 0], [0, max_height - 1 + 0]], dtype="float32")
    M = cv2.getPerspectiveTransform(np.float32(pts), dst)
    print('image shape: ', image.shape)
    print('bboxes shape: ', bboxes.shape)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    # new_bboxes = cv2.warpPerspective(bboxes, M, (max_width, max_height))
    out = np.dot(M,(pts[2][0],pts[2][1],1))
    out = out/out[2]
    # print('before, after: ', pts[2], dst[2], out)
    # exit()
    new_bboxes = []
    for bbox in bboxes:
        new_bbox = []
        for point in bbox:
            new_point = np.dot(M,(point[0],point[1],1))
            new_bbox.append(list(new_point[:2]/new_point[2]))
        new_bboxes.append(new_bbox)
    return warped, new_bboxes

def read_labelme_corner(label_filepath):
    rs = [None, None, None, None]
    data = json.load(open(label_filepath, 'r'))
    list_object = data['shapes']
    bbox = []
    for oj in list_object:
        if oj['shape_type'] == 'rectangle':
            tl = oj['points'][0]
            br = oj['points'][1]
            pt = [int((oj['points'][0][0] + oj['points'][1][0])/2) , int((oj['points'][0][1] + oj['points'][1][1])/2)]
            if oj['label'] not in ['0', '1', '2', '3']:
                continue
            label = int(oj['label'])
            if rs[label] == None:
                rs[label] = pt
            else:
                return None
    for i in rs:
        if i == None:
            return None
    return rs

def create_labelme_object(pointss,labels,image_path):
    image = Image.open(image_path)
    width, height = image.size
    print(width, height)

    list_object = []
    for points,label in zip(pointss,labels):
        list_object.append(create_object(points,label))
    return {
        "version": "4.5.10",
        "flags": {},
        "shapes": list_object,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

def create_object(points, label):
    return {
      "label": label,
      "points": points,
      "group_id": None,
      "shape_type": "polygon",
      "flags": {}
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source', required=True,
        help='segment obejct json path', default='data/source/corner.json'
    )
    parser.add_argument(
        '--source-image', required=True,
        help='segment obejct json path', default='data/source/corner.json'
    )
    parser.add_argument(
        '--dest', required=True,
        help='new ocr object json path after conver keypoint', default='data/dest/tmp'
    )

    args = parser.parse_args()

    if os.path.exists(args.dest) is False:
        os.makedirs(args.dest, exist_ok=True)

    print(args.source)
    filenames = [f for f in os.listdir(args.source) if os.path.splitext(f)[1] == '.json']
    
    
    for idx, filename in enumerate(filenames):
        print(filename)
        label_path_corner = os.path.join(args.source, filename)
        print(idx, label_path_corner)
        image_path = os.path.join(args.source_image, filename.replace('.json', '.jpg'))

        seg = read_labelme_corner(label_path_corner)
        if seg == None:
            continue

        json_data = create_labelme_object([seg], ['0'], image_path)
        # print(json_data)
        out_label_path = os.path.join(args.dest, filename)
        with open(out_label_path, 'w') as f_json:
            json.dump(json_data, f_json)

        