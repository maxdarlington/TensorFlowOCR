import os
from PIL import Image
import numpy as np
from sklearn.utils import shuffle

class DatasetLoader:
    def __init__(self, data_dir, base_dir):
        self.data_dir = data_dir
        self.save_dir = os.path.join(os.path.dirname(base_dir), "content", "data")
        
    def preprocess_image(self, image_path):
        try:
            # Load image
            img = Image.open(image_path)
            
            if img.size != (28, 28):
                # Resize if not already 28x28
                print(f"Resizing image {image_path} from {img.size} to (28, 28)")
                img = img.resize((28, 28), Image.Resampling.BILINEAR)
            else:
                print(f"Image {image_path} is already 28x28")

            # Convert to grayscale
            img = img.convert('L')
                        
            # Convert to numpy array and normalize to float value between 0 - 1
            img_array = np.array(img) / 255.0   

            return img_array
        
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def load_dataset(self, data_dir):
        images = []
        labels = []
        
        # Walk through the data directory
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    
                    # Get label from directory name
                    label = os.path.basename(root)
                    
                    # Preprocess image
                    processed_image = self.preprocess_image(image_path)
                    
                    if processed_image is not None:
                        images.append(processed_image)
                        labels.append(label)
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)

        print(f"Loaded {len(images)} images")
        print(f"Image shape: {images.shape}")
        print(f"Unique labels: {set(labels)}")

        # Shuffle the dataset
        images_sh, labels_sh = shuffle(images, labels, random_state=42)
        print("Finished shuffling images and labels")
        print("Do you want to save the processed data? (y/n): ")
        save_choice = input().strip().lower()
        if save_choice == 'y':
            self.save_processed_data(images, labels, self.save_dir)
        else:
            print("Processed data not saved.")
        return images_sh, labels_sh
    
    def save_processed_data(self, images, labels, save_dir):
        try:
            filename = input("Enter name for processed data: ")
            filepath = os.path.join(save_dir, f"{filename}.npz")
            
            # Save both image array and metadata
            np.savez(filepath,
                    images=images,
                    labels=labels,
                    images_shape=images.shape,
                    labels_shape=labels.shape
                    )            
            print(f"Processed data saved to {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            return None
        
    def load_processed_data(self, filename):
        try:
            data = np.load(filename)
            return {
                'images': data['images'],
                'labels': data['labels'],
                'images_shape': data['images_shape'],
                'labels_shape': data['labels_shape']
            }

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None