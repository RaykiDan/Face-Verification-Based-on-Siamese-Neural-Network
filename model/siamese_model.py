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

def verify(model, detection_threshold=0.5, verification_threshold=0.5, max_validation=10):
    input_path = os.path.join('application_data', 'input_image', 'input_image.jpg')
    verification_dir = os.path.join('data', 'positive')

    print("[INFO] Verifying image...")

    try:
        input_img = preprocess(input_path)
    except Exception as e:
        print("[ERROR] Gagal load input image:", e)
        return [], False

    all_files = [f for f in os.listdir(verification_dir) if f.endswith('.jpg') or f.endswith('.png')]
    if len(all_files) == 0:
        print("[ERROR] Tidak ada gambar valid di folder positive.")
        return [], False

    # Pilih beberapa gambar acak untuk divalidasi
    selected_files = random.sample(all_files, min(max_validation, len(all_files)))

    results = []
    for image_name in selected_files:
        try:
            validation_img = preprocess(os.path.join(verification_dir, image_name))
            result = model.predict(
                [tf.expand_dims(input_img, 0), tf.expand_dims(validation_img, 0)],
                verbose=0
            )
            results.append(result)
        except Exception as e:
            print(f"[WARNING] Gagal proses {image_name}: {e}")

    if not results:
        return [], False

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(results) > verification_threshold

    print(f"[INFO] {detection}/{len(results)} lolos detection_threshold")
    return results, verification

def load_model():
    return tf.keras.models.load_model('model/siamese_model.h5',
        custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})