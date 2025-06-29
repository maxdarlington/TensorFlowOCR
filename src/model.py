import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Input

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
        self.model.add(Input(shape=(28, 28, 1)))
        self.model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'))
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
        """Calculate optimal number of epochs based on dataset size"""
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
        """Get custom number of epochs from user with proper error handling"""
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
                raise
            except EOFError:
                print("\n[WARNING] End of input detected.")
                raise
    
    def result(self, test_images, test_labels, idx):
        # Use same encoding as training
        str_labels = [str(label) for label in test_labels]
        valid_indices = [i for i, label in enumerate(str_labels) if label in self.valid_labels]
        
        # Check if we have any valid labels
        if not valid_indices:
            print("No valid labels found in test data")
            return 0.0
            
        # Filter images and encode labels using the same valid_indices
        filtered_images = test_images[valid_indices]
        encoded_labels = np.array([self.label_to_idx[str_labels[i]] for i in valid_indices])
        
        # Check if idx is within the filtered data range
        if idx >= len(filtered_images):
            print(f"Index {idx} is out of range. Max index: {len(filtered_images)-1}")
            return None, None
        
        # Test single prediction
        image = filtered_images[idx:idx+1]  # Keep batch dimension
        label = encoded_labels[idx:idx+1]
            
        test_loss, test_accuracy = self.model.evaluate(image, label, verbose=0)
        predicted_label = self.predict(filtered_images[idx])
        actual_label = str_labels[valid_indices[idx]]
        correct = actual_label == predicted_label
        confidence = float(test_accuracy)
        
        # Return result as a dictionary for CSV saving
        return {
            'index': idx,
            'actual_label': actual_label,
            'predicted_label': predicted_label,
            'correct': correct,
            'confidence': confidence
        }
        
    
    def predict(self, image):
        # Ensure image has the correct shape (batch_size, height, width, channels)
        if image.ndim == 2:
            # Image is (28, 28) - add batch and channel dimensions
            image = image.reshape(1, 28, 28, 1)
        elif image.ndim == 3:
            # Image is (28, 28, 1) - add batch dimension
            image = image.reshape(1, *image.shape)
        elif image.ndim == 4:
            # Image is already (batch_size, 28, 28, 1) - check if batch_size is 1
            if image.shape[0] != 1:
                raise ValueError(f"Expected single image, got batch of size {image.shape[0]}")
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        predictions = self.model.predict(image, verbose=0)
        predicted_idx = np.argmax(predictions[0])  # Use index 0 for single prediction
        predicted_label = self.idx_to_label[predicted_idx]
        return predicted_label
    
    def plot(self, image, predicted_label):
        plt.imshow(image.reshape(28, 28), cmap='gray')
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