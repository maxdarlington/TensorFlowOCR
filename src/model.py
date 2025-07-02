import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Input
import os

class Model:
    def __init__(self):
        """Initialize the OCR model, label mappings, and data augmentation."""
        # Create consistent label mapping
        nums = [str(i) for i in range(10)]  # 0-9
        upper_chars = [chr(i) for i in range(65, 91)]  # A-Z
        lower_chars = [chr(i) for i in range(97, 123)]  # a-z
        symbols = ['!', '@', '#', '$', '%', '&', '*', '+', '-', '?', '<', '>']
        self.valid_labels = nums + upper_chars + lower_chars + symbols

        # Create a fixed mapping dictionary
        self.label_to_idx = {label: idx for idx, label in enumerate(self.valid_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # Build the CNN model
        self.model = Sequential()
        self.model.add(Input(shape=(28, 28, 1)))
        self.model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=len(self.valid_labels), activation='softmax'))

        # Data augmentation
        self.datagen = ImageDataGenerator(
            rotation_range=20,  # ±20 degrees
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest'
        )

    def train(self, images_sh, labels_sh):
        """Train the model on the provided images and labels."""
        # Convert labels to proper format and encode them
        str_labels = [str(label) for label in labels_sh]
        encoded_labels = np.array([self.label_to_idx[label] for label in str_labels if label in self.valid_labels])
        # Filter corresponding images to match valid labels
        valid_indices = [i for i, label in enumerate(str_labels) if label in self.valid_labels]
        filtered_images = images_sh[valid_indices]
        print("\n" + "-" * 40)
        print("TRAINING CONFIGURATION")
        print("-" * 40)
        # Calculate optimal epochs based on dataset size
        dataset_size = len(filtered_images)
        optimal_epochs = self._calculate_optimal_epochs(dataset_size)
        print(f"[INFO] Dataset size: {dataset_size:,} samples")
        print(f"[INFO] Optimal epochs: {optimal_epochs}")
        print("-" * 40)
        print("1. Use optimal epochs (recommended)")
        print("2. Enter custom number of epochs")
        print("-" * 40)
        while True:
            try:
                choice = input("Select option (1-2): ").strip()
                if choice == '1':
                    epoch_input = optimal_epochs
                    print(f"[INFO] Using optimal epochs: {epoch_input}")
                    break
                elif choice == '2':
                    epoch_input = self._get_custom_epochs()
                    break
                else:
                    print("[ERROR] Please enter 1 or 2.")
                    continue
            except KeyboardInterrupt:
                print("\n[WARNING] Operation cancelled by user.")
                print("Returning to main menu...")
                return None
            except EOFError:
                print("\n[WARNING] End of input detected.")
                return None
        try:
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            print(f"\n[INFO] Starting training with {epoch_input} epochs...")
            history = self.model.fit(filtered_images, encoded_labels, validation_split=0.2, 
                            batch_size=16, epochs=epoch_input, verbose=1)
            print("[SUCCESS] Training completed.")
            self.model.summary()
            return history
        except Exception as e:
            print(f"[ERROR] Error during training: {e}")
            return None

    def _calculate_optimal_epochs(self, dataset_size):
        """Calculate optimal number of epochs based on dataset size."""
        if dataset_size < 1000:
            return 20
        elif dataset_size < 5000:
            return 15
        elif dataset_size < 10000:
            return 12
        elif dataset_size < 20000:
            return 10
        else:
            return 8

    def _get_custom_epochs(self):
        """Get custom number of epochs from user with proper error handling."""
        while True:
            try:
                epoch_input = input("Enter the number of learning cycles for training: ").strip()
                if not epoch_input:
                    print("[ERROR] Please enter a number.")
                    continue
                epoch_input = int(epoch_input)
                if epoch_input <= 0:
                    print("[ERROR] Please enter a positive number.")
                    continue
                if epoch_input > 100:
                    print(f"[WARNING] {epoch_input} epochs is quite high. This may take a long time.")
                    confirm = input("Continue anyway? (y/n): ").strip().lower()
                    if confirm not in ['y', 'yes']:
                        continue
                return epoch_input
            except ValueError:
                print(f"[ERROR] Invalid input: '{epoch_input}'. Please enter a number.")
                continue
            except KeyboardInterrupt:
                print("\n[WARNING] Operation cancelled by user.")
                print("Returning to main menu...")
                return None
            except EOFError:
                print("\n[WARNING] End of input detected.")
                raise

    def result(self, test_images, test_labels, idx, filename=None):
        """Evaluate a single test image and return prediction details."""
        str_labels = [str(label) for label in test_labels]
        valid_indices = [i for i, label in enumerate(str_labels) if label in self.valid_labels]
        if not valid_indices:
            print("No valid labels found in test data")
            return 0.0
        filtered_images = test_images[valid_indices]
        encoded_labels = np.array([self.label_to_idx[str_labels[i]] for i in valid_indices])
        if idx >= len(filtered_images):
            print(f"Index {idx} is out of range. Max index: {len(filtered_images)-1}")
            return None, None
        image = filtered_images[idx:idx+1]  # Keep batch dimension
        label = encoded_labels[idx:idx+1]
        # Get prediction probabilities
        preds = self.model.predict(image, verbose=0)
        pred_idx = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds[0]))
        predicted_label = self.idx_to_label[pred_idx]
        actual_label = str_labels[valid_indices[idx]]
        correct = actual_label == predicted_label
        return {
            'index': idx,
            'input': os.path.basename(filename) if filename else None,
            'actual_label': actual_label,
            'predicted_label': predicted_label,
            'correct': correct,
            'confidence': confidence
        }

    def predict(self, image):
        """Predict the label for a single image."""
        # Ensure image has the correct shape (batch_size, height, width, channels)
        if image.ndim == 2:
            image = image.reshape(1, 28, 28, 1)
        elif image.ndim == 3:
            image = image.reshape(1, 28, 28, 1)
        preds = self.model.predict(image, verbose=0)
        pred_idx = np.argmax(preds, axis=1)[0]
        return self.idx_to_label[pred_idx]

    def plot(self, image, predicted_label):
        """Plot the image with its predicted label."""
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f'Predicted: {predicted_label}')
        plt.axis('off')
        plt.show()

    def save_model(self, model_path):
        """Save the trained model to the specified path."""
        self.model.save(model_path)

    def load_model(self, model_path):
        """Load a model from the specified path."""
        self.model = tf.keras.models.load_model(model_path)