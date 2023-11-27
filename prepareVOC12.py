import os
from PIL import Image
import numpy as np
import cv2


# img size
img_size = (224, 224)

class_dict = {
    'background':   [0, 0, 0],
    'aeroplane':    [128, 0, 0],
    'bicycle':      [0, 128, 0],
    'bird':         [128, 128, 0],
    'boat':         [0, 0, 128],
    'bottle':       [128, 0, 128],
    'bus':          [0, 128, 128],
    'car':          [128, 128, 128],
    'cat':          [64, 0, 0],
    'chair':        [192, 0, 0],
    'cow':          [64, 128, 0],
    'diningtable':  [192, 128, 0],
    'dog':          [64, 0, 128],
    'horse':        [192, 0, 128],
    'motorbike':    [64, 128, 128],
    'person':       [192, 128, 128],
    'pottedplant':  [0, 64, 0],
    'sheep':        [128, 64, 0],
    'sofa':         [0, 192, 0],
    'train':        [128, 192, 0],
    'tvmonitor':    [0, 64, 128]
}

single_channel_dict = {
    (0, 0, 0): 0,
    (128, 0, 0): 1,
    (0, 128, 0): 2,
    (128, 128, 0): 3,
    (0, 0, 128): 4,
    (128, 0, 128): 5,
    (0, 128, 128): 6,
    (128, 128, 128): 7,
    (64, 0, 0): 8,
    (192, 0, 0): 9,
    (64, 128, 0): 10,
    (192, 128, 0): 11,
    (64, 0, 128): 12,
    (192, 0, 128): 13,
    (64, 128, 128): 14,
    (192, 128, 128): 15,
    (0, 64, 0): 16,
    (128, 64, 0): 17,
    (0, 192, 0): 18,
    (128, 192, 0): 19,
    (0, 64, 128): 20,
}

def load_data(data_dir, labels_dir, img_size=(224, 224)):
    data = []
    labels = []  
    for folder in [data_dir, labels_dir]:
        for root, _, files in os.walk(folder):
            for file in files:
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
                
                if folder == data_dir:
                    img = img.astype(np.float32) / 255.0
                    data.append(img)
                else:
                    labels.append(img)  
    return np.array(data), np.array(labels)

def label_to_onehot(label_map, class_dict):
    one_hot_map = np.zeros((*label_map.shape[:-1], len(class_dict)), dtype=np.uint8)
    for i, color in enumerate(class_dict.values()):
        if color in label_map:
            mask = np.all(label_map == np.array(color), axis=-1)
            one_hot_map[..., i] = mask.astype(np.uint8)
    return one_hot_map

def label_to_onehot_ResUnet(label_map, class_dict):
    one_hot_map = np.zeros((*label_map.shape[:-1], len(class_dict)), dtype=np.uint8)
    for i, color in enumerate(class_dict.values()):
        if np.any(np.all(label_map == color, axis=-1)):
            mask = np.all(label_map == np.array(color), axis=-1)
            one_hot_map[..., i] = mask.astype(np.uint8)
    return one_hot_map

def onehot_to_label(onehot_map, class_dict):
    h, w = onehot_map.shape[:2]
    label_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    max_indices = np.argmax(onehot_map, axis=-1)

    for i, color in enumerate(class_dict.values()):
        label_map[max_indices == i] = color

    return label_map

def batch_onehot_to_label(onehot_maps, class_dict):
    batch_size, h, w, _ = onehot_maps.shape
    label_maps = np.zeros((batch_size, h, w, 3), dtype=np.uint8)
    
    for b in range(batch_size):
        onehot_map = onehot_maps[b]
        max_indices = np.argmax(onehot_map, axis=-1)

        for i, color in enumerate(class_dict.values()):
            label_maps[b, max_indices == i] = color

    return label_maps


def rgb_to_single_channel(label_map, single_channel_dict):
    single_channel_label_map = np.zeros((label_map.shape[0], label_map.shape[1]), dtype=np.uint8)

    for color, idx in single_channel_dict.items():
        color_array = np.array(color).reshape(1, 1, 3)
        mask = np.all(label_map == color_array, axis=-1)
        single_channel_label_map[mask] = idx

    return single_channel_label_map

def single_channel_to_rgb(label_map, single_channel_dict):
    rgb_label_map = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)

    for color, idx in single_channel_dict.items():
        mask = label_map == idx
        rgb_label_map[mask] = color

    return rgb_label_map

# # set up paths
# data_dir = 'C:\\Users\\22428\\Desktop\\CS6420 AdvCompVision\\Semester_Project_Hao_Lan\\VOCdevkit\\VOC2012\\'
# image_dir = os.path.join(data_dir, 'JPEGImages')
# train_list_file = os.path.join(data_dir, 'ImageSets', 'Segmentation', 'train.txt')
# val_list_file = os.path.join(data_dir, 'ImageSets', 'Segmentation', 'val.txt')
# seg_class_dir = os.path.join(data_dir, 'SegmentationClass')

# # create folders of processed dataset
# output_dir = 'VOC12'
# train_dir = os.path.join(output_dir, 'train')
# val_dir = os.path.join(output_dir, 'val')
# train_label_dir = os.path.join(output_dir, 'train_label')
# val_label_dir = os.path.join(output_dir, 'val_label')

# # create directories for output
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(val_dir, exist_ok=True)
# os.makedirs(train_label_dir, exist_ok=True)
# os.makedirs(val_label_dir, exist_ok=True)

# # read train list
# with open(train_list_file, 'r') as f:
#     train_list = f.read().splitlines()

# # read validation list
# with open(val_list_file, 'r') as f:
#     val_list = f.read().splitlines()

# # process training set
# for img_name in train_list:
#     img_path = os.path.join(image_dir, img_name + '.jpg')
#     label_path = os.path.join(seg_class_dir, img_name + '.png')
#     img = Image.open(img_path)
#     label = Image.open(label_path)

#     # resize image and label
#     img = img.resize(img_size)
#     label = label.resize(img_size)

#     # save image and label
#     img.save(os.path.join(train_dir, img_name + '.jpg'))
#     label.save(os.path.join(train_label_dir, img_name + '.png'))

# # process validation set
# for img_name in val_list:
#     img_path = os.path.join(image_dir, img_name + '.jpg')
#     label_path = os.path.join(seg_class_dir, img_name + '.png')
#     img = Image.open(img_path)
#     label = Image.open(label_path)

#     # resize image and label
#     img = img.resize(img_size)
#     label = label.resize(img_size)

#     # save image and label
#     img.save(os.path.join(val_dir, img_name + '.jpg'))
#     label.save(os.path.join(val_label_dir, img_name + '.png'))
