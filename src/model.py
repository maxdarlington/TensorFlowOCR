import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class Model:
    def __init__(self):
        self.model = Sequential()
        # First convolutional block
        self.model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1), padding='same'))
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
        self.label_dict = {}  # Add dictionary to store label mappings
        
        # Create consistent label mapping
        nums = [str(i) for i in range(10)]  # 0-9
        chars = [chr(i) for i in range(65, 91)]  # A-Z
        self.valid_labels = nums + chars
        # Create a fixed mapping dictionary
        self.label_to_idx = {label: idx for idx, label in enumerate(self.valid_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def train(self, images_sh, labels_sh):
        # Convert labels to proper format and encode them
        str_labels = [str(label).upper() for label in labels_sh]
        encoded_labels = np.array([self.label_to_idx[label] for label in str_labels if label in self.valid_labels])
        
        # Filter corresponding images to match valid labels
        valid_indices = [i for i, label in enumerate(str_labels) if label in self.valid_labels]
        filtered_images = images_sh[valid_indices]
        
        epoch_input = int(input("Enter the number of learning cycles for training: "))
        try:
            if epoch_input <= 0:
                print("Invalid input. Defaulting to 10.")
                epoch_input = 10
        except ValueError:
            print("Invalid input. Please enter a number. Defaulting to 10.")
            epoch_input = 10

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(filtered_images, encoded_labels, validation_split=0.2, 
                           batch_size=16, epochs=epoch_input, verbose=1)
        print("Training completed.")
        self.model.summary()
        return history
    
    def accuracy(self, test_images, test_labels):
        # Use same encoding as training
        str_labels = [str(label).upper() for label in test_labels]
        encoded_labels = np.array([self.label_to_idx[label] for label in str_labels if label in self.valid_labels])
        valid_indices = [i for i, label in enumerate(str_labels) if label in self.valid_labels]
        filtered_images = test_images[valid_indices]
        
        test_loss, test_accuracy = self.model.evaluate(filtered_images, encoded_labels)
        print(f"Test Accuracy: {test_accuracy}")
        print(f"Test Loss: {test_loss}")
        return test_accuracy
    
    def plot_prediction(self, test_images, index):
        preds = self.model.predict(test_images)
        pred_idx = np.argmax(preds[index])
        predicted_label = self.idx_to_label[pred_idx]
        
        plt.imshow(test_images[index].reshape(28, 28), cmap='gray')
        plt.title(f"Predicted Label: {predicted_label}")
        plt.axis('off')
        plt.show()

    def save_model(self, model_path):
        if not model_path.endswith('.keras'):
            model_path += '.keras'
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")