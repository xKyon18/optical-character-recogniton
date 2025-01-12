import tensorflow as tf
import numpy as np


mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = (X_train / 255) - 0.5
X_test = (X_test / 255) - 0.5


X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, 3, input_shape=(28, 28, 1), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

model.fit(
    X_train,
    tf.keras.utils.to_categorical(y_train),
    epochs=5,
    batch_size=100,
    validation_data=(X_test, tf.keras.utils.to_categorical(y_test))
)

model.save_weights('cnn.h5')
