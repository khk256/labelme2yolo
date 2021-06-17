# labelme2yolo
This is simple python code for converting labelme annotation file to yolo darknet format

# How to use

```bash

python labelme2yolo.py --path /path/to/labelme --output /path/to/yolo --object dog,cat,cow --ratio 0.9

```

```
--path : path to labelme annotation

--output : path to save converted yolo train data

--object : define which object to convert

--ratio : define ration for splitting train, valid
```