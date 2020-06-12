import os
from PIL import Image
import cv2
import numpy as np

DATA_PATH = '/home/duyngoc/project/Attention_RNN/Dataset/base_data'
NEW_DATA_PATH = '/home/duyngoc/project/Attention_RNN/Dataset/train_attention'
txt_files = [x for x in os.listdir(DATA_PATH) if x.endswith('.txt')]
img_files = [x for x in os.listdir(DATA_PATH) if x.endswith('.jpg')]

for filename in txt_files[:50]:
    txt_file = open(os.path.join(DATA_PATH, filename), 'r', encoding="utf8")
    name = filename.replace('.txt', '')
    image_name = filename.replace('.txt', '.jpg')
    img = Image.open(os.path.join(DATA_PATH, image_name)).convert("RGB")
    opencv_image = np.array(img)
    for index, line in enumerate(txt_file):
        info = line.split(',')
        x_min = int(info[0])
        x_max = int(info[2])
        y_min = int(info[1])
        y_max = int(info[5])
        text = ', '.join(info[8:])
        f = open(os.path.join(NEW_DATA_PATH, name + "-crop-" + str(index) + ".txt"), "w+")
        print(os.path.join(NEW_DATA_PATH, name + "-crop-" + str(index) + ".txt"))
        f.write("%s" % text)
        img_crop = opencv_image[y_min:y_max, x_min:x_max]
        cv2.imwrite(os.path.join(NEW_DATA_PATH, name + "-crop-" + str(index) + ".jpg"), img_crop)
        f.close()




