import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Gunakan non-GUI backend (headless backend)
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import confusion_matrix
from model.siamese_model import build_siamese_model
from utils.preprocessing import preprocess
from utils import config
import random

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 16

# Load dataset paths
def get_data():
    anchor_images = [os.path.join(config.ANC_PATH, f) for f in os.listdir(config.ANC_PATH)]
    positive_images = [os.path.join(config.POS_PATH, f) for f in os.listdir(config.POS_PATH)]
    negative_images = [os.path.join(config.NEG_PATH, f) for f in os.listdir(config.NEG_PATH)]
    return anchor_images, positive_images, negative_images

# Data generator
def data_generator(anchor_images, positive_images, negative_images):
    while True:
        anchors = random.sample(anchor_images, BATCH_SIZE)
        positives = random.sample(positive_images, BATCH_SIZE // 2)
        negatives = random.sample(negative_images, BATCH_SIZE // 2)

        X1, X2, y = [], [], []

        for a, p in zip(anchors[:len(positives)], positives):
            X1.append(preprocess(a))
            X2.append(preprocess(p))
            y.append(1)

        for a, n in zip(anchors[len(positives):], negatives):
            X1.append(preprocess(a))
            X2.append(preprocess(n))
            y.append(0)

        yield (tf.stack(X1), tf.stack(X2)), tf.convert_to_tensor(y)

# Training function
def train():
    anchor_images, positive_images, negative_images = get_data()
    steps_per_epoch = min(len(anchor_images), len(positive_images), len(negative_images)) // (BATCH_SIZE // 2)

    model = build_siamese_model()
    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    # History
    history_loss = []
    history_acc = []
    history_precision = []
    history_recall = []

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0
        y_true_all, y_pred_all = [], []

        train_gen = data_generator(anchor_images, positive_images, negative_images)

        for step in range(steps_per_epoch):
            (X1, X2), y_true = next(train_gen)
            with tf.GradientTape() as tape:
                y_pred = model([X1, X2], training=True)
                loss = loss_fn(y_true, y_pred)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            total_loss += loss.numpy()

            y_true_all.extend(y_true.numpy())
            y_pred_all.extend(y_pred.numpy().reshape(-1))

        # Metrics per epoch
        y_pred_binary = np.array(y_pred_all) > 0.5
        acc = np.mean(y_pred_binary == y_true_all)
        precision = Precision()(y_true_all, y_pred_binary).numpy()
        recall = Recall()(y_true_all, y_pred_binary).numpy()
        cm = confusion_matrix(y_true_all, y_pred_binary)

        print(f"Loss: {total_loss/steps_per_epoch:.4f} | Acc: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        print("Confusion Matrix:\n", cm)

        # Simpan history
        history_loss.append(total_loss/steps_per_epoch)
        history_acc.append(acc)
        history_precision.append(precision)
        history_recall.append(recall)

    # Simpan model
    model.save(config.MODEL_PATH)
    print(f"Model disimpan ke {config.MODEL_PATH}")

    # Plot grafik
    plot_training(EPOCHS, history_loss, history_acc, history_precision, history_recall)

# Plot function
def plot_training(epochs, loss, acc, precision, recall):
    epochs_range = range(1, epochs+1)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, loss, marker='o')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, acc, marker='o')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, precision, marker='o')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')

    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, recall, marker='o')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')

    plt.tight_layout()
    plt.show()

    plt.tight_layout()
    plt.savefig('training_result.png')
    print("Grafik training telah disimpan di: training_result.png")

if __name__ == "__main__":
    train()
