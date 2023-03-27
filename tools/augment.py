import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import cv2
from PIL import Image
import json
import argparse
import os
import numpy as np
import shutil
# https://github.com/aleju/imgaug
ia.seed(4)
np.random.seed(0)

from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

X = get_truncated_normal(mean=0, sd=15, low=-45, upp=45)

MAX_LENGTH = 100000

def read_labelme(label_filepath):
    data = json.load(open(label_filepath, 'r'))
    
    bboxes = []
    labels = []
    list_object = data['shapes']
    for oj in list_object:
        if oj['shape_type'] == 'polygon':
            bboxes.append(oj['points'])
            labels.append(oj['label'])
        elif oj['shape_type'] == 'rectangle':
            tl = oj['points'][0]
            br = oj['points'][1]
            bboxes.append([tl, [br[0], tl[1]], br, [tl[0], br[1]]])
            labels.append(oj['label'])
    return bboxes, labels
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


def augment(angle):
    seq = iaa.Sequential([
        # iaa.Affine(rotate=(-25, 25), mode=['edge', 'constant', 'symmetric', 'reflect', 'wrap'], fit_output=True),
        iaa.Affine(rotate=angle, mode=['edge', 'constant'], fit_output=True),
        # iaa.Affine(scale=0.5)
        iaa.AdditiveGaussianNoise(scale=(0, 10)),
        # iaa.Crop(percent=(0, 0.2))
    ])
    return seq

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
    parser.add_argument(
        '--csv', required=True,
        help='Label file', default='data/dest/tmp.csv'
    )
    parser.add_argument(
        '--loop', type=int,
        help='number of loop', default=1
    )
    args = parser.parse_args()

    if os.path.exists(args.dest) is False:
        os.makedirs(args.dest, exist_ok=True)
        
    f = open(args.csv, 'w')

    print(args.source)
    filenames = [f for f in os.listdir(args.source) if os.path.splitext(f)[1] == '.jpg']


    # angles = [0]*(1*args.loop*len(filenames)) + list(X.rvs(MAX_LENGTH))
    # idxs = np.random.randint(4, size=MAX_LENGTH)
    # angles = [int(angles[i]+idxs[i]*90)  for i in range(MAX_LENGTH)]
    # angles = [i for i in range(0,360)] + [360+i if i < 0 else i for i in angles]
    angles = 90*np.random.randint(4, size=MAX_LENGTH)

    # print(angles)
    print(max(angles), min(angles))
    # exit()
    
    idx = 0
    for i in range(5*args.loop):
        for filename in filenames:
            print(filename)
            filename, file_extension = os.path.splitext(filename)
            file_path = os.path.join(args.source, filename+file_extension)
            print(i, file_path)
            # out_dir = os.path.join(args.dest, str(angles[idx]))
            # if os.path.exists(out_dir) is False:
            #     os.makedirs(out_dir, exist_ok=True)
            label = str(angles[idx])
            name_idx = 0
            while os.path.exists(os.path.join(args.dest, \
                    filename + '_' + str(name_idx) + '_' + label +file_extension)) is True:
                name_idx += 1
            filename = filename + '_' + str(name_idx) + '_' + label +file_extension
            # out_path = os.path.join(out_dir, filename)

            image = cv2.imread(file_path)
            print(image.shape)
            h,w,_ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            kps = KeypointsOnImage([
                Keypoint(x=0, y=0),
                Keypoint(x=w, y=0),
                Keypoint(x=w, y=h),
                Keypoint(x=0, y=h)
            ], shape=image.shape)
            seq = augment(angles[idx])
            idx+=1
            images_aug, kps_aug = seq(image=image,keypoints=kps)
            tmp = []
            for i in range(len(kps.keypoints)):
                after = kps_aug.keypoints[i]
                tmp.append([int(after.x), int(after.y)])
            kps_aug = tmp

            transformed_image = cv2.cvtColor(images_aug, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(out_path, transformed_image)
            cv2.imwrite(os.path.join(args.dest, filename), transformed_image)
            label_path = file_path.replace('jpg', 'json')
            out_label_path = os.path.join(args.dest, filename).replace('jpg', 'json')
            print(os.path.dirname(out_label_path))
            if os.path.exists(os.path.dirname(out_label_path)) is False:
                os.makedirs(os.path.dirname(out_label_path), exist_ok=True)
            shutil.copy(label_path, \
                        out_label_path)

            json_data = create_labelme_object([kps_aug], [label], os.path.join(args.dest, filename))
            # print(json_data)
            with open(os.path.join(args.dest,filename).replace('jpg', 'json'), 'w') as f_json:
                json.dump(json_data, f_json)

            print(label_path)
            bboxes, labels = read_labelme(label_path)
            kps_augs = []
            for bboxe, label in zip(bboxes, labels):
                # print('bboxe:',bboxe)
                # print('label:',label)
                kps = KeypointsOnImage([
                    Keypoint(x=b[0], y=b[1]) for b in bboxe
                ], shape=image.shape)
                images_aug, kps_aug = seq(image=image,keypoints=kps)
                tmp = []
                for i in range(len(kps.keypoints)):
                    after = kps_aug.keypoints[i]
                    tmp.append([int(after.x), int(after.y)])
                kps_aug = tmp
                kps_augs.append(kps_aug)
            json_data = create_labelme_object(kps_augs, labels, os.path.join(args.dest, filename))
            # print(json_data)
            with open(out_label_path, 'w') as f_json:
                json.dump(json_data, f_json)

            f.write(filename + ',' + label + '\n')
    f.close()

