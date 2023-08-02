
# data/captcha_click
# 总计有1032对样本，分为451个不同的字符；其中只出现一次的有269个，而出现最多的一个字有24次
"""
    裁切图片 为siamese构建数据集
"""
import json
import os
from collections import defaultdict
import numpy as np
import cv2
import matplotlib.pyplot as plt


def crop_gtbox(xyxy, img_file):
    image = cv2.imread(img_file)
    x1, y1, x2, y2 = xyxy
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

data_file = 'data/captcha_click/captcha_click'
file_list = os.listdir(data_file)

# 孪生网络的训练数据存放路径
save_path = 'datasets/data_siamese'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 保存宽高信息用于分析
w_list = []
h_list = []
data_dict = {'char': defaultdict(list), 'target': defaultdict(list)}
count = 0
for file in file_list:
    if '.json' in file:
        file_path = os.path.join(data_file, file)
        file_name = file.split('.')[0]
        with open(file_path, 'r', encoding='utf-8') as f:
            img_labels = json.load(f)
        img_path = file_path.replace('.json', '.jpg')
        shapes = img_labels['shapes']
        x1_list = []
        char_list = []
        for item in shapes:
            text = item['text']
            text_unicode = text.encode('unicode_escape').decode().replace('\\', '')
            label = item['label']
            # 裁切出的图片的命名格式
            new_img_name = f'{label}_{text_unicode}_{file_name}.jpg'
            save_file = os.path.join(save_path, new_img_name)
            if label == 'char':
                data_dict['char'][text].append(new_img_name)
            elif label == 'target':
                data_dict['target'][text].append(new_img_name)
            count += 1
            points = item['points']
            np_data = np.array(points, dtype=np.int64)
            w = np_data[1][0] - np_data[0][0]
            h = np_data[1][1] - np_data[0][1]
            w_list.append(w)
            h_list.append(h)
            xyxy = np_data.reshape((-1,))
            # 裁切图片并保存
            crop_img = crop_gtbox(xyxy, img_path)
            cv2.imwrite(save_file, crop_img)
        
with open('datasets/data_annotation.json', 'w', encoding='utf-8') as f:
    json.dump(data_dict, f, indent=2, ensure_ascii=False)

print(f'总计样本{count}个, 分为不同的字符char{len(data_dict["char"])}个, target{len(data_dict["target"])}')

num_char_list = [len(data_dict["char"][i]) for i in data_dict["char"].keys()]
# 可视化
# 宽高分布
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(w_list, h_list)
plt.xlabel('Width')
plt.ylabel('Height')

# 字符数量分布统计
plt.subplot(122)
data = np.array(num_char_list)
x, y = np.unique(data, return_counts=True)
plt.bar(range(len(x)), y)
plt.xticks(range(len(x)), x)
plt.xlabel('character repetitions')
plt.ylabel('count')
plt.show()