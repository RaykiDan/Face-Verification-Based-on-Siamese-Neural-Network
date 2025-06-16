import numpy as np
import random
import os
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, Dense, Input, Flatten
from tensorflow.keras.models import Model

# Custom L1 distance layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# Embedding model
def make_embedding():
    input = Input(shape=(100, 100, 3))
    x = Conv2D(64, (10, 10), activation='relu')(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (7, 7), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (4, 4), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, (4, 4), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(4096, activation='sigmoid')(x)
    return Model(inputs=input, outputs=x)

# Siamese model
def build_siamese_model():
    input_img = Input(name='input_img', shape=(100, 100, 3))
    validation_img = Input(name='validation_img', shape=(100, 100, 3))

    embedding = make_embedding()

    input_embedding = embedding(input_img)
    validation_embedding = embedding(validation_img)

    l1_layer = L1Dist()
    distances = l1_layer(input_embedding, validation_embedding)
    output = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_img, validation_img], outputs=output)

def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

# def contains_face(image_path):
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     return len(faces) > 0

def verify(model, detection_threshold=0.5, verification_threshold=0.5, return_identity=False):
    input_path = os.path.join('application_data', 'input_image', 'input_image.jpg')
    try:
        input_img = preprocess(input_path)
    except Exception as e:
        print("[ERROR] Gagal load input image:", e)
        return [], None if return_identity else False

    matched_user = None
    best_score = 0

    for user in os.listdir('data'):
        user_path = os.path.join('data', user, 'positive')
        if not os.path.isdir(user_path):
            continue
        all_imgs = [img for img in os.listdir(user_path) if img.endswith(('.jpg', '.png'))]
        if not all_imgs:
            continue

        results = []
        for img_name in all_imgs[:10]:  # batasi 10 untuk efisiensi
            try:
                validation_img = preprocess(os.path.join(user_path, img_name))
                result = model.predict([tf.expand_dims(input_img, 0), tf.expand_dims(validation_img, 0)], verbose=0)
                results.append(result)
            except Exception as e:
                print(f"[WARNING] Error loading {img_name}: {e}")

        if not results:
            continue

        detection = np.sum(np.array(results) > detection_threshold)
        score = detection / len(results)

        if score > verification_threshold and score > best_score:
            best_score = score
            matched_user = user

        # if not contains_face(input_path):
        #     print("[INFO] Tidak ada wajah terdeteksi.")
        #     return [], None

        print(f"[DEBUG] {user} confidence scores: {np.round(results, 2).reshape(-1).tolist()}")

    if return_identity:
        return [], matched_user
    else:
        return [], matched_user is not None

def load_model():
    return tf.keras.models.load_model('model/siamese_model.h5',
        custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})