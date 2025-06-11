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
                    )            
            print(f"Processed data saved to {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            return None
        
    def load_processed_data(self, filename):
        try:
            data = np.load(filename)
            images = data['images']
            labels = data['labels']
            return images, labels

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None, None

    def dataDirCheck(self, data_dir):
        # Get available test data directories
        print("Available datasets:")
        data_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if not data_dirs:
            print("No datasets found.")
            return None, None
            
        for i, dir_name in enumerate(data_dirs):
            print(f"{i+1}. {dir_name}")
        
        # Select test directory
        try:
            data_idx = int(input("Select dataset number: ")) - 1
            if data_idx < 0 or data_idx >= len(data_dirs):
                print("Invalid selection. Exiting test mode.")
                return None, None
            selected_test_dir = os.path.join(data_dir, data_dirs[data_idx])
        except ValueError:
            print("Invalid input. Please enter a number.")
            return None, None
        
        print(f"Loading data from {data_dirs[data_idx]}...")
        images, labels = self.load_dataset(selected_test_dir)  
        return images, labels
    
    def npzCheck(self, data_dir):
        print(f"Checking for .npz files in: {data_dir}")
        
        # Verify directory exists
        if not os.path.exists(data_dir):
            print(f"Error: Directory {data_dir} does not exist")
            return None, None
        
        # Get available test data files with debug info
        files = [file for file in os.listdir(data_dir) if file.endswith('.npz')]
        print(f"Found {len(files)} .npz files")
        
        if not files:
            print("No .npz files found.")
            return None, None
            
        # Print available files with full paths
        for i, filename in enumerate(files):
            full_path = os.path.join(data_dir, filename)
            print(f"{i+1}. {filename} ({full_path})")
        
        # Select test file
        try:
            idx = int(input("Select dataset number: ")) - 1
            if idx < 0 or idx >= len(files):
                print(f"Invalid selection. Index {idx+1} out of range (1-{len(files)})")
                return None, None
                
            selected_file = os.path.join(data_dir, files[idx])
            print(f"Loading data from: {selected_file}")
            
            # Try loading the file
            try:
                data = np.load(selected_file)
                if 'images' not in data or 'labels' not in data:
                    print(f"Error: File missing required arrays 'images' or 'labels'")
                    print(f"Available arrays: {data.files}")
                    return None, None
                    
                images = data['images']
                labels = data['labels']
                
                print(f"Successfully loaded:")
                print(f"- Images shape: {images.shape}")
                print(f"- Labels shape: {labels.shape}")
                
                return images, labels
                
            except Exception as e:
                print(f"Error loading npz file: {str(e)}")
                return None, None
                
        except ValueError:
            print("Invalid input. Please enter a number.")
            return None, None