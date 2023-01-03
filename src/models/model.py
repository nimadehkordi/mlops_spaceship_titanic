from tensorflow import keras
import tensorflow as tf
import numpy as np

#create a model class with a constructor and a method
class Model:
    def __init__(self, input_shape):
        self.model_sequential = None
        self.input_shape = input_shape
        self.create_model_sequential()

    def create_model_sequential(self):
        # Build the sequential model
        self.model_sequential = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.input_shape,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation='sigmoid'),
        ])

        # Print a summary of the model
        self.model_sequential.summary()

        # Compile the model
        self.model_sequential.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss='binary_crossentropy',
                    metrics=['accuracy'],
                    )
