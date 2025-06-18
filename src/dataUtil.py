import os
from PIL import Image
import numpy as np
from sklearn.utils import shuffle
import multiprocessing as mp
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob

def process_single_image(image_path):
    """Standalone function for processing a single image (for multiprocessing)"""
    try:
        # Load and process image in one go
        with Image.open(image_path) as img:
            # Convert to grayscale and resize in one operation if needed
            if img.size != (28, 28):
                img = img.resize((28, 28), Image.Resampling.LANCZOS).convert('L')
            else:
                img = img.convert('L')
            
            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            return img_array, os.path.basename(os.path.dirname(image_path))
    
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None, None

def process_image_batch(image_paths):
    """Process a batch of images for multiprocessing"""
    results = []
    for image_path in image_paths:
        img_array, label = process_single_image(image_path)
        if img_array is not None:
            results.append((img_array, label))
    return results

class DatasetLoader:
    def __init__(self, data_dir, base_dir):
        self.data_dir = data_dir
        self.save_dir = os.path.join(os.path.dirname(base_dir), "content", "data")
        
    def process_image(self, image_path):
        """Optimized single image processing"""
        try:
            # Load and process image in one go
            with Image.open(image_path) as img:
                # Convert to grayscale and resize in one operation if needed
                if img.size != (28, 28):
                    img = img.resize((28, 28), Image.Resampling.LANCZOS).convert('L')
                else:
                    img = img.convert('L')
                
                # Convert to numpy array and normalize
                img_array = np.array(img, dtype=np.float32) / 255.0
                return img_array
        
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def get_all_image_paths(self, data_dir):
        """Get all image paths efficiently using glob"""
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
        image_paths = []
        
        for ext in image_extensions:
            pattern = os.path.join(data_dir, '**', ext)
            image_paths.extend(glob.glob(pattern, recursive=True))
        
        return image_paths
    
    def load_dataset(self, data_dir):
        """Optimized dataset loading with multiprocessing"""
        print("Scanning for image files...")
        image_paths = self.get_all_image_paths(data_dir)
        
        if not image_paths:
            print("No image files found!")
            return None, None
        
        print(f"Found {len(image_paths)} images. Processing...")
        start_time = time.time()
        
        # Try multiprocessing first, fallback to single-threaded if it fails
        try:
            # Determine optimal number of processes
            num_processes = min(mp.cpu_count(), 8)  # Cap at 8 processes to avoid overhead
            batch_size = max(1, len(image_paths) // (num_processes * 4))  # Ensure reasonable batch sizes
            
            print(f"Using {num_processes} processes with batch size {batch_size}")
            
            # Split image paths into batches
            batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
            
            all_results = []
            
            # Process batches in parallel
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(process_image_batch, batch): batch 
                    for batch in batches
                }
                
                # Collect results with progress tracking
                completed = 0
                for future in as_completed(future_to_batch):
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    completed += 1
                    
                    # Progress update every 10% or every 10 batches
                    if completed % max(1, len(batches) // 10) == 0 or completed % 10 == 0:
                        progress = (completed / len(batches)) * 100
                        elapsed = time.time() - start_time
                        print(f"Progress: {progress:.1f}% ({completed}/{len(batches)} batches) - {elapsed:.1f}s elapsed")
        
        except Exception as e:
            print(f"Multiprocessing failed: {e}")
            print("Falling back to single-threaded processing...")
            
            # Fallback to single-threaded processing
            all_results = []
            for i, image_path in enumerate(image_paths):
                img_array, label = process_single_image(image_path)
                if img_array is not None:
                    all_results.append((img_array, label))
                
                # Progress update every 100 images
                if (i + 1) % 100 == 0:
                    progress = ((i + 1) / len(image_paths)) * 100
                    elapsed = time.time() - start_time
                    print(f"Progress: {progress:.1f}% ({i + 1}/{len(image_paths)} images) - {elapsed:.1f}s elapsed")
        
        # Separate images and labels
        if not all_results:
            print("No images were successfully processed!")
            return None, None
        
        images, labels = zip(*all_results)
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels)
        
        total_time = time.time() - start_time
        print(f"Processed {len(images)} images in {total_time:.2f} seconds")
        print(f"Average time per image: {total_time/len(images)*1000:.2f} ms")
        print(f"Image shape: {images.shape}")
        print(f"Unique labels: {sorted(set(labels))}")

        # Shuffle the data
        print("Shuffling data...")
        images_sh, labels_sh = shuffle(images, labels, random_state=42)
        print("Data shuffling complete")
        
        print("Do you want to save the processed data? (y/n): ")
        save_choice = input().strip().lower()
        if save_choice == 'y':
            self.save_processed_data(images, labels, self.save_dir)
        else:
            print("Processed data not saved.")
        
        return images_sh, labels_sh
    
    def save_processed_data(self, images, labels, save_dir):
        while True:
            try:
                filename = input("Enter name for processed data: ").strip()
                                
                if not filename:
                    print("Filename cannot be empty. Please try again.")
                    continue
                
                invalid_chars = '<>:"/\\|?*'
                if any(char in filename for char in invalid_chars):
                    print(f"Filename contains invalid characters. Please avoid: {invalid_chars}")
                    continue
                
                filepath = os.path.join(save_dir, f"{filename}.npz")
                
                if os.path.exists(filepath):
                    overwrite = input("File already exists. Overwrite? (y/n): ").strip().lower()
                    if overwrite != 'y':
                        continue
                
                if images is None or labels is None:
                    print("Error: Images or labels data is None")
                    return None
                
                np.savez(filepath,
                        images=images,
                        labels=labels,
                        )   
                        
                print(f"Processed data saved to {filepath}")
                return filepath 
            
            except Exception as e:
                print(f"Error saving data: {e}")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    print("Operation cancelled.")
                    return None
                continue
        
    def load_processed_data(self, filename):
        try:
            data = np.load(filename)
            images = data['images']
            labels = data['labels']
            return images, labels

        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None

    def dataDirCheck(self, data_dir):
        # get available data directories
        print("Available datasets:")
        data_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if not data_dirs:
            print("No datasets found.")
            return None, None
            
        for i, dir_name in enumerate(data_dirs):
            print(f"{i+1}. {dir_name}")
        
        # select data directory
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
        
        # verify directory exists
        if not os.path.exists(data_dir):
            print(f"Error: Directory {data_dir} does not exist")
            return None, None
        
        # get available test data files with debug info
        files = [file for file in os.listdir(data_dir) if file.endswith('.npz')]
        print(f"Found {len(files)} .npz files")
        
        if not files:
            print("No .npz files found.")
            return None, None
            
        # print available files with full paths
        for i, filename in enumerate(files):
            full_path = os.path.join(data_dir, filename)
            print(f"{i+1}. {filename} ({full_path})")
        
        # select test file
        try:
            idx = int(input("Select dataset number: ")) - 1
            if idx < 0 or idx >= len(files):
                print(f"Invalid selection. Index {idx+1} out of range (1-{len(files)})")
                return None, None
                
            selected_file = os.path.join(data_dir, files[idx])
            print(f"Loading data from: {selected_file}")
            
            # load the file
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
                print(f"Error loading npz file: {e}")
                return None, None
                
        except ValueError:
            print("Invalid input. Please enter a number.")
            return None, None