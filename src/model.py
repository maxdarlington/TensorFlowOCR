import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class Model:
    def __init__(self):
        self.model = Sequential()
        # First convolutional block
        self.model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', 
                             input_shape=(28,28,1), padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        
        # Second convolutional block
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        
        # Third convolutional block
        self.model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        
        # Fully connected layers
        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=36, activation='softmax'))
        self.le = LabelEncoder()

    def train(self, images_sh, labels_sh):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(images_sh, labels_sh, validation_split=0.2, batch_size=16, epochs=10)
        return history
    
    def predict(self, test_images):
        preds = self.model.predict(test_images)
        predicted_labels = self.le.inverse_transform(np.argmax(preds, axis=1))
        return predicted_labels

    def plot_prediction(self, test_images, index):
        predicted_labels = self.predict(test_images)
        plt.imshow(test_images[index].reshape(28, 28), cmap='gray')
        plt.title(f"Predicted Label: {predicted_labels[index]}")
        plt.axis('off')
        plt.show()

    def accuracy(self, test_images, test_labels):
        encoded_labels = self.le.fit_transform(test_labels)
        test_loss, test_accuracy = self.model.evaluate(test_images, encoded_labels)
        print(f"Test Accuracy: {test_accuracy}")
        print(f"Test Accuracy: {test_loss}")
        return test_accuracy