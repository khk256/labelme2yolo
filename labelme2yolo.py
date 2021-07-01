import os
import json
import shutil
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

# labelme_box(x1,y1,x2,y2) to yolo_box(cx,cy,w,h)
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

im_ext = ['.png', '.jpg', '.gif']

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

# train.txt, valid.txt
yolo_train_txt = open(yolo_train_text_path, 'w')
yolo_valid_txt = open(yolo_valid_text_path, 'w')

obj_classes = []

for labelme_json_path in glob(f'{args.path}/*.json'): # json 만 조회

    labelme_annotation_open = open(labelme_json_path, 'r')
    labelme_annotation = json.load(labelme_annotation_open)
    
    # train, valid split
    train_or_valid = 'train' if random() < args.ratio else 'valid'
    yolo_annotation_name = labelme_json_path.split('/')[-1:][0].replace('.json', '.txt')
    yolo_annotation_path = os.path.join(args.output, train_or_valid, yolo_annotation_name)

    # write to train.txt
    im_file_list = [yolo_annotation_name.split('.')[0] + ext for ext in im_ext]
    im_file_name = [im_file for im_file in im_file_list if im_file in os.listdir(args.path)][0]
    if train_or_valid == 'train':
        yolo_train_txt.write(os.path.join(args.output, train_or_valid, im_file_name + '\n'))
    else:
        yolo_valid_txt.write(os.path.join(args.output, train_or_valid, im_file_name + '\n'))
    
    for shapes in labelme_annotation['shapes']:

        if args.object:
            obj_classes = str(args.object).split(',')
        else:
            if shapes['label'] in obj_classes:
                pass
            else:
                obj_classes.append(shapes['label'])

        for obj_class in obj_classes:
            if obj_class in shapes['label']:

                obj_label = shapes['label']
                xmin = min(float(shapes['points'][0][0]), float(shapes['points'][1][0])) # x1, x2
                xmax = max(float(shapes['points'][0][0]), float(shapes['points'][1][0])) # x1, x2
                ymin = min(float(shapes['points'][0][1]), float(shapes['points'][1][1])) # y1, y2
                ymax = max(float(shapes['points'][0][1]), float(shapes['points'][1][1])) # y1, y2

                im = Image.open(os.path.join(args.path, labelme_annotation['imagePath']))
                # convert((w, h), bbox)
                bbox = convert((int(im.size[0]), int(im.size[1])), (xmin, xmax, ymin, ymax))
                
                yolo_annotation_file = open(yolo_annotation_path, 'w')
                yolo_annotation_file.write(f'{obj_classes.index(obj_class)} {" ".join([str(a) for a in bbox])} \n')
                shutil.copy(os.path.join(args.path, im_file_name), os.path.join(args.output, train_or_valid, im_file_name))
            else:
                pass

    labelme_annotation_open.close()
    yolo_annotation_file.close()

# write to obj.data
yolo_obj_data_file = open(yolo_obj_data_path, 'w')
yolo_obj_data_file.write('classes = ' + str(len(obj_classes)) + '\n')
yolo_obj_data_file.write('train = ' + yolo_train_text_path + '\n')
yolo_obj_data_file.write('valid = ' + yolo_valid_path + '\n')
yolo_obj_data_file.write('names = ' + yolo_obj_names_path + '\n')
yolo_obj_data_file.write('backup = ' + yolo_backup_path)
yolo_obj_data_file.close()

# write to obj.names
yolo_obj_names_file = open(yolo_obj_names_path, 'w')
if args.object:
    for obj in str(args.object).split(','):
        yolo_obj_names_file.write(obj + '\n')
else:
    for obj in obj_classes:
        yolo_obj_names_file.write(obj + '\n')

yolo_obj_names_file.close()
yolo_train_txt.close()
yolo_valid_txt.close()

print('Done!!')