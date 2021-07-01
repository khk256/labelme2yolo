# labelme2yolo

This is simple python script to convert labelme annotation file into yolo darknet dataset format.

# How to use

## Example : 
```bash
python labelme2yolo.py --path /path/to/labelme --output /path/to/yolo --object dog,cat,cow --ratio 0.9
```
---
```
--path : type path to labelme labelme data directory, must be image and json pairs

--output : path to save converted yolo training dataset (train, train.txt, valid, valid.txt, obj.data, obj.names)

--object : define which object to convert, empty it for all object to be converted

--ratio : define ratio for splitting train, valid data. default : 0.9
```
