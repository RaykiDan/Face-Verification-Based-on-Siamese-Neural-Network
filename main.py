import tensorflow as tf
import os
from utils.config import POS_PATH, NEG_PATH, ANC_PATH, MODEL_PATH
from utils.preprocessing import preprocess_twin
from model.siamese_model import build_siamese_model
from model.training import train
import numpy as np

# Load dataset
def load_data():
    anchor = tf.data.Dataset.list_files(ANC_PATH + '/*.jpg').take(800)
    positive = tf.data.Dataset.list_files(POS_PATH + '/*.jpg').take(800)
    negative = tf.data.Dataset.list_files(NEG_PATH + '/*.jpg').take(800)

    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(800))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(800))))

    data = positives.concatenate(negatives)
    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=10000)
    data = data.batch(16)
    data = data.prefetch(8)
    return data

# Main function
def main():
    print("Loading data...")
    data = load_data()

    print("Building model...")
    model = build_siamese_model()

    print("Training model...")
    train(model, data, epochs=10)

    print("Saving model...")
    model.save(MODEL_PATH, save_format="keras")

if __name__ == '__main__':
    main()
