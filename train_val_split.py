import os
import csv
from shutil import move
from pycocotools.coco import COCO

val_dir = "./data/images/val"
train_dir = "./data/images/train"
train_ids_csv = "TrainImageIds.csv"
split_size = 0.8
coco = COCO('./data/annotations/captions_train2014.json')

if not os.path.isdir(val_dir):
    os.makedirs(val_dir)

with open(train_ids_csv, "r", newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    img_ids =list(csvreader)[0]

print(img_ids[0])

split_id = int(split_size*len(img_ids))
# train_ids = img_ids[0:split_id]
val_ids = img_ids[split_id:]

for idx in val_ids:
    path = coco.loadImgs(img_id)[0]['file_name']
    move(train_dir+path, val_dir+path)