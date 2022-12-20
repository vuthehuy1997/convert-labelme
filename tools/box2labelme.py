
# import tqdm
import os
import json
from PIL import Image

def xywh2xyxy(points):
    x,y,w,h = points
    print(x,y,w,h)
    x1 = x - w/2
    x2 = x + w/2
    y1 = y - h/2
    y2 = y + h/2
    print([x1,y1,x2,y2])
    return [x1,y1,x2,y2]
def create_labelme_object(list_xywh_nom, list_label, image_path):
    image = Image.open(image_path)
    width, height = image.size
    print(width, height)
    list_xyxy = []
    for xywh_nom in list_xywh_nom:

        xywh_nom[0] *= width
        xywh_nom[1] *= height
        xywh_nom[2] *= width
        xywh_nom[3] *= height
                
        xyxy = xywh2xyxy(xywh_nom)
        xyxy = list(map(int, xyxy))
        list_xyxy.append(xyxy)
        print('xyxy: ', xyxy)
    list_object = []
    for xyxy, label in zip(list_xyxy,list_label):
        list_object.append(create_object(xyxy,label))
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
      "points": [
        [
          points[0],
          points[1]
        ],
        [
          points[2],
          points[3]
        ]
      ],
      "group_id": None,
      "shape_type": "rectangle",
      "flags": {}
    }

path = 'visualize_index/public_test'

os.makedirs(path, exist_ok=True)

txt_files = sorted(os.listdir(path)) # change to right directory
txt_files = [i for i in txt_files if i.endswith(".txt")]
# print(txt_files)
# exit()
for file_name in (txt_files):
    print(file_name)
    lines = open(os.path.join(path,file_name), "r")
    list_xywh_nom = []
    list_label = []
    for line in lines:
        line = line.split()
        print('line: ', line)
        points = list(map(float, line[1:5]))
        label = line[0]
        print('points: ', points)
        print('label: ', label)
        list_xywh_nom.append(points)
        list_label.append(label)
    json_data = create_labelme_object(list_xywh_nom, list_label, os.path.join(path,file_name).replace('txt', 'jpg'))
    print(json_data)
    with open(os.path.join(path,file_name).replace('txt', 'json'), 'w') as f:
        json.dump(json_data, f)
    # exit()
