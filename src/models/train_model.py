import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from model import Model

def main() -> int:
    # Set random seed for reproducibility
    np.random.seed(0)
    tf.random.set_seed(0)

    # Load the train and validation data
    data = pd.read_csv('./data/raw/train.csv')
    data.drop(['Name', 'Cabin'], axis=1, inplace=True)
    data.dropna(inplace=True) # drop rows with null values

    data.CryoSleep = data.CryoSleep.astype('int64')	# convert CryoSleep to int64
    data.VIP = data.VIP.astype('int64')	# convert VIP to int64
    data.Transported = data.Transported.astype('int64')	# convert Transported to int64

    # Convert categorical data to numerical data
    data.HomePlanet = data.HomePlanet.astype('category')
    data['HomePlanet'] = data.HomePlanet.cat.codes

    data.Destination = data.Destination.astype('category')
    data['Destination'] = data.Destination.cat.codes

    # Shuffle rows
    data = data.sample(frac=1).reset_index(drop=True)

    # Split data into features and labels for training and validation
    train_data, val_data = train_test_split(data, test_size=0.2)
    train_features = train_data.drop(['Transported'], axis=1)
    train_labels = train_data['Transported']
    val_features = val_data.drop(['Transported'], axis=1)
    val_labels = val_data['Transported']

    # Convert data to numpy arrays
    train_features = np.array(train_features).astype('float32')	
    train_labels = np.array(train_labels).astype('float32')
    val_features = np.array(val_features).astype('float32')
    val_labels = np.array(val_labels).astype('float32')

    # Create the model
    model = Model(train_features.shape[1])

    history = model.model_sequential.fit(train_features, train_labels, epochs=100, validation_data=(val_features, val_labels), batch_size=32)

    model.model_sequential.save('./models/model.h5')

    # Plot training & validation accuracy values and save to file
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./reports/figures/accuracy.png')
    
    return 0

if __name__ == '__main__':
    exit(main())