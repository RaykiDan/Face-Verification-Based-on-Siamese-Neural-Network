import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam

loss_fn = BinaryCrossentropy()
optimizer = Adam(1e-4)
precision = Precision()
recall = Recall()

@tf.function
def train_step(model, batch):
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]
        yhat = model(X, training=True)
        loss = loss_fn(y, yhat)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(model, train_data, epochs=10):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for batch in train_data:
            loss = train_step(model, batch)
        print(f"Loss: {loss.numpy():.4f}")
