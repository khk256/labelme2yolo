import os
import json
import cv2
import imutils
import argparse
from glob import glob
from PIL import Image
from random import random

parser = argparse.ArgumentParser(description='Convert Labelme annotation to Yolo darknet format')

def arg_directory(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f'`{path}` is not valid')

parser.add_argument('--path', 
                    type = arg_directory,
                    help = 'Directory to labelme annotation',
                    default=None
                    )

parser.add_argument('--output',
                    type = arg_directory,
                    help = 'Path to save yolo',
                    default = None 
                    )

parser.add_argument('--object',
                    type = str,
                    help = 'Type object class',
                    default = False
                    )

parser.add_argument('--ratio',
                    type = float,
                    help = 'Training ratio',
                    default = 0.9
                    )

args = parser.parse_args()

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

yolo_train_path = os.path.join(args.output, 'train')
yolo_valid_path = os.path.join(args.output, 'valid')
yolo_train_text_path = os.path.join(args.output, 'train.txt')
yolo_valid_text_path = os.path.join(args.output, 'valid.txt')
yolo_obj_data_path = os.path.join(args.output, 'obj.data')
yolo_obj_names_path = os.path.join(args.output, 'obj.names')
yolo_backup_path = os.path.join(args.output, 'backup')

# train, valid 디렉토리 생성
os.mkdir(yolo_train_path)
os.mkdir(yolo_valid_path)

for labelme_json_path in glob(f'{args.path}/*.json'): # json 만 조회

    labelme_annotation_open = open(labelme_json_path, 'r')
    labelme_annotation = json.load(labelme_annotation_open)

    for shapes in labelme_annotation['shapes']:

        if args.object:
            obj_class = args.object
        else:
            obj_class = ''

        if obj_class in shapes['label']:

            obj_label = shapes['label']
            xmin = min(float(shapes['points'][0][0]), float(shapes['points'][1][0])) # x1, x2
            xmax = max(float(shapes['points'][0][0]), float(shapes['points'][1][0])) # x1, x2
            ymin = min(float(shapes['points'][0][1]), float(shapes['points'][1][1])) # y1, y2
            ymax = max(float(shapes['points'][0][1]), float(shapes['points'][1][1])) # y1, y2

            im = Image.open(os.path.join(args.path, labelme_annotation['imagePath']))
            # convert((w, h), bbox)
            bbox = convert((int(im.size[0]), int(im.size[1])), (xmin, xmax, ymin, ymax))

            yolo_annotation_file.write(f'{obj_label} {" ".join([str(a) for a in bbox])} \n')

    labelme_annotation_open.close()