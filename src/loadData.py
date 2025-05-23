import os
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

class DatasetLoader:
    def __init__(self, data_dir):
        """
        Initialize the DatasetLoader with the data directory path.
        
        Args:
            data_dir (str): Path to the directory containing the dataset
        """
        self.data_dir = data_dir
        
    def preprocess_image(self, image_path):
        """
        Preprocess a single image to be 28x28 grayscale.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image as a 28x28 grayscale array
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Convert to grayscale
            img = img.convert('L')
            
            # Resize to 28x28
            img = img.resize((28, 28), Image.BILINEAR)
            
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img) / 255.0
            
            return img_array
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def load_dataset(self):
        """
        Load all images from the data directory and preprocess them.
        
        Returns:
            tuple: (images, labels) where images is a numpy array of preprocessed images
                  and labels is a list of corresponding labels
        """
        images = []
        labels = []
        
        # Walk through the data directory
        for root, dirs, files in os.walk(self.data_dir):
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
        
        # Encode labels from string to integer
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        print("Finished encoding labels")

        # Shuffle the dataset
        images_sh, labels_sh = shuffle(images, labels, random_state=42)
        print("Finished shuffling images and labels")

        return images_sh, labels_sh