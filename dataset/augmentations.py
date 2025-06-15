import tensorflow as tf
import numpy as np
import uuid
import os
import cv2

def data_aug(img):
    data = []
    for _ in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1, 2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1, 3))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100,
                                                     seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9, upper=1,
                                                   seed=(np.random.randint(100), np.random.randint(100)))
        data.append(img)
    return data

def augment_folder(input_folder):
    for file_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, file_name)
        img = cv2.imread(img_path)
        if img is not None:
            img_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)
            augmented_images = data_aug(img_tensor)
            for image in augmented_images:
                save_path = os.path.join(input_folder, f"{uuid.uuid1()}.jpg")
                cv2.imwrite(save_path, image.numpy())
