
"""
    转yolo格式, 划分训练验证集
"""
import json
import os
import random
import shutil


def xyxy2xywh(xyxy, img_size):
    img_w, img_h = img_size
    (x1, y1), (x2, y2) = xyxy
    x_c = (x1 + x2) / 2 / img_w
    y_c = (y1 + y2) / 2 / img_h
    h = (y2 - y1) / img_h
    w = (x2 - x1) / img_w
    return x_c, y_c, w, h


data_dir = 'data/captcha_click/captcha_click'
train_dir = 'datasets/train'
val_dir = 'datasets/val'
save_path = 'data/labels'
for folder in [train_dir, val_dir, save_path]:
    if not os.path.exists(folder):
        os.makedirs(folder)
split_ratio = 0.9  # 0.9 for training, 0.1 for validation
class_dict = {'target': 0, 'char': 1}

# 转yolo格式
for img_name in os.listdir(data_dir):
    if '.jpg' in img_name:
        annotation = os.path.join(data_dir, img_name.replace('.jpg', '.json'))
        with open(annotation, 'r', encoding='utf-8') as f:
            img_labels = json.load(f)
        img_h, img_w = (img_labels['imageHeight'], img_labels['imageWidth'])
        shapes = img_labels['shapes']

        save_label = os.path.join(save_path, img_name.replace('.jpg', '.txt'))
        f = open(save_label, 'w', encoding='utf-8')
        box_list = []
        for shape in shapes:
            label = shape['label']
            class_index = class_dict[label]
            points = shape['points']
            x_c, y_c, w, h = xyxy2xywh(points, (img_w, img_h))
            box_list.append(f'{class_index} {x_c} {y_c} {w} {h}')
        f.write('\n'.join(box_list))
        f.close()

# 划分训练集
label_list = os.listdir(save_path)
random.seed(100)
random.shuffle(label_list)
for set in [train_dir, val_dir]:
    for folder in ['labels/', 'images/']:
        path = os.path.join(set, folder)
        if not os.path.exists(path):
            os.mkdir(path)
# 训练集
train_set = label_list[:int(len(label_list)*split_ratio)]
print('train set num:', len(train_set))
for item in train_set:
    label_file = os.path.join(save_path, item)
    img_file = os.path.join(data_dir, item.replace('.txt', '.jpg'))
    train_label_path = os.path.join(train_dir, 'labels/' + item)
    train_img_path = os.path.join(train_dir, 'images/' + item.replace('.txt', '.jpg'))
    shutil.copy(label_file, train_label_path)
    shutil.copy(img_file, train_img_path)
# 验证集
val_set = label_list[int(len(label_list)*split_ratio):]
print('val set num:', len(val_set))
for item in val_set:
    label_file = os.path.join(save_path, item)
    img_file = os.path.join(data_dir, item.replace('.txt', '.jpg'))
    train_label_path = os.path.join(val_dir, 'labels/' + item)
    train_img_path = os.path.join(val_dir, 'images/' + item.replace('.txt', '.jpg'))
    shutil.copy(label_file, train_label_path)
    shutil.copy(img_file, train_img_path)