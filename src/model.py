import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense

class Model:
    def __init__(self):
        # Create consistent label mapping
        nums = [str(i) for i in range(10)]  # 0-9
        upper_chars = [chr(i) for i in range(65, 91)]  # A-Z
        lower_chars = [chr(i) for i in range(97, 123)]  # a-z
        symbols = ['!', '@', '#', '$', '%', '&', '*', '+', '-', '?', '<', '>']
        self.valid_labels = nums + upper_chars + lower_chars + symbols

        # Create a fixed mapping dictionary
        self.label_to_idx = {label: idx for idx, label in enumerate(self.valid_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
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
        self.model.add(Dense(units=len(self.valid_labels), activation='softmax'))

        # Data augmentation
        self.datagen = ImageDataGenerator(
            rotation_range=20,  # +-20 degrees
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest'
        )

    def train(self, images_sh, labels_sh):
        # Convert labels to proper format and encode them
        str_labels = [str(label) for label in labels_sh]
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

        try:
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            history = self.model.fit(filtered_images, encoded_labels, validation_split=0.2, 
                            batch_size=16, epochs=epoch_input, verbose=1)
            print("Training completed.")
            self.model.summary()
            return history
        
        except Exception as e:
            print(f"Error during training: {e}")
            return None
    
    def accuracy(self, test_images, test_labels):
        # Use same encoding as training
        str_labels = [str(label) for label in test_labels]
        valid_indices = [i for i, label in enumerate(str_labels) if label in self.valid_labels]
        
        # Check if we have any valid labels
        if not valid_indices:
            print("No valid labels found in test data")
            return 0.0
            
        encoded_labels = np.array([self.label_to_idx[label] for label in str_labels if label in self.valid_labels])
        filtered_images = test_images[valid_indices]
        
        # Ensure we have matching data
        if len(filtered_images) != len(encoded_labels):
            print("Mismatch between images and labels")
            return 0.0
        
        # Ensure we have some data to evaluate
        if len(filtered_images) == 0:
            print("No valid test data available")
            return 0.0
            
        test_loss, test_accuracy = self.model.evaluate(filtered_images, encoded_labels)
        print(f"Test Accuracy: {test_accuracy:.2%}")
        print(f"Test Loss: {test_loss:.2f}")
        return test_accuracy, test_loss
    
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