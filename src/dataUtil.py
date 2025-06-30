import os
from PIL import Image
import numpy as np
from sklearn.utils import shuffle
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import csv
from main import show_loading_throbber

def save_results_to_csv(results, csv_path):
    """Save a list of result dictionaries to a CSV file."""
    if not results:
        print("No results to save.")
        return
    keys = results[0].keys()
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {csv_path}")

def foldername_to_label(folder_name):
    if folder_name.startswith('upper'):
        return folder_name[-1].upper()
    elif folder_name.startswith('lower'):
        return folder_name[-1].lower()
    symbol_map = {
        'at': '@', 'exclmark': '!', 'hash': '#', 'dollar': '$', 'percent': '%',
        'ampersand': '&', 'asterisk': '*', 'plus': '+', 'minus': '-', 'quesmark': '?',
        'lessthan': '<', 'greaterthan': '>'
    }
    if folder_name in symbol_map:
        return symbol_map[folder_name]
    return folder_name  # for digits

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
            
            # Convert to numpy array, normalize, and add channel dimension
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = img_array.reshape(28, 28, 1)  # Add channel dimension for grayscale
            folder_name = os.path.basename(os.path.dirname(image_path))
            label = foldername_to_label(folder_name)
            return img_array, label
    
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
                
                # Convert to numpy array, normalize, and add channel dimension
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = img_array.reshape(28, 28, 1)  # Add channel dimension for grayscale
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
        print("[INFO] Preparing to process dataset. This may take a while...")
        show_loading_throbber("Processing dataset", duration=1.0)
        print("\n" + "-" * 40)
        print("DATASET PROCESSING")
        print("-" * 40)
        print("[INFO] Scanning for image files...")
        
        image_paths = self.get_all_image_paths(data_dir)
        
        if not image_paths:
            print("[ERROR] No image files found in the specified directory!")
            return None, None
        
        print(f"[INFO] Found {len(image_paths):,} images. Processing...")
        start_time = time.time()

        # Use single-threaded processing for small datasets
        if len(image_paths) <= 1000:
            print("[INFO] Small dataset detected (<=100 images). Using single-threaded processing.")
            all_results = []
            for i, image_path in enumerate(image_paths):
                img_array, label = process_single_image(image_path)
                if img_array is not None:
                    all_results.append((img_array, label))
                # Progress update every 10 images
                if (i + 1) % 10 == 0 or (i + 1) == len(image_paths):
                    progress = ((i + 1) / len(image_paths)) * 100
                    elapsed = time.time() - start_time
                    print(f"[INFO] Progress: {progress:.1f}% ({i + 1:,}/{len(image_paths):,} images) - {elapsed:.1f}s elapsed")
        else:
            # Try multiprocessing first, fallback to single-threaded if it fails
            try:
                # Conservative multiprocessing configuration for system-friendly operation
                # Use only 2 processes maximum to reduce system load
                num_processes = min(2, mp.cpu_count() // 2)  # Use at most 2 processes or half of CPU cores
                if num_processes < 1:
                    num_processes = 1
                
                # Larger batch sizes to reduce overhead and memory pressure
                batch_size = max(50, len(image_paths) // (num_processes * 2))  # Larger batches, fewer total
                print(f"[INFO] Using {num_processes} process(es) with batch size {batch_size}")
                
                # Split image paths into batches
                batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
                all_results = []
                
                # Process batches in parallel with limited concurrency
                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    # Submit batches one at a time to limit memory usage
                    futures = []
                    completed = 0
                    
                    for i, batch in enumerate(batches):
                        # Submit batch
                        future = executor.submit(process_image_batch, batch)
                        futures.append(future)
                        
                        # Wait for completion if we have too many pending futures
                        # This prevents memory buildup and reduces system load
                        if len(futures) >= num_processes:
                            # Wait for one future to complete
                            done_future = futures.pop(0)
                            try:
                                batch_results = done_future.result()
                                all_results.extend(batch_results)
                                completed += 1
                            except Exception as e:
                                print(f"[WARNING] Batch processing failed: {e}")
                            
                            # Progress update
                            if completed % max(1, len(batches) // 10) == 0 or completed % 5 == 0:
                                progress = (completed / len(batches)) * 100
                                elapsed = time.time() - start_time
                                print(f"[INFO] Progress: {progress:.1f}% ({completed:,}/{len(batches):,} batches) - {elapsed:.1f}s elapsed")
                    
                    # Wait for remaining futures
                    for future in futures:
                        try:
                            batch_results = future.result()
                            all_results.extend(batch_results)
                            completed += 1
                        except Exception as e:
                            print(f"[WARNING] Batch processing failed: {e}")
                        
                        # Final progress updates
                        if completed % max(1, len(batches) // 10) == 0 or completed % 5 == 0:
                            progress = (completed / len(batches)) * 100
                            elapsed = time.time() - start_time
                            print(f"[INFO] Progress: {progress:.1f}% ({completed:,}/{len(batches):,} batches) - {elapsed:.1f}s elapsed")
                            
            except Exception as e:
                print(f"[WARNING] Multiprocessing failed: {e}")
                print("[INFO] Falling back to single-threaded processing...")
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
                        print(f"[INFO] Progress: {progress:.1f}% ({i + 1:,}/{len(image_paths):,} images) - {elapsed:.1f}s elapsed")
        
        # Separate images and labels
        if not all_results:
            print("[ERROR] No images were successfully processed!")
            return None, None
        
        images, labels = zip(*all_results)
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels)
        
        total_time = time.time() - start_time
        print("\n" + "-" * 40)
        print("PROCESSING COMPLETE")
        print("-" * 40)
        print(f"[INFO] Processed {len(images):,} images in {total_time:.2f} seconds")
        print(f"[INFO] Average time per image: {total_time/len(images)*1000:.2f} ms")
        print(f"[INFO] Image shape: {images.shape}")
        
        # Convert numpy string objects to regular strings for clean display
        unique_labels = sorted(set(str(label) for label in labels))
        print(f"[INFO] Unique labels: {unique_labels}")

        # Shuffle the data
        print("\n[INFO] Shuffling data...")
        show_loading_throbber("Shuffling data", duration=0.5)
        images_sh, labels_sh = shuffle(images, labels, random_state=42)
        print("[SUCCESS] Data shuffling complete")
        
        print("\n[INFO] Do you want to save the processed data? (y/n): ")
        save_choice = input().strip().lower()
        if save_choice == 'y':
            self.save_processed_data(images, labels, self.save_dir)
        else:
            print("[INFO] Processed data not saved.")
        
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
        print("\n" + "-" * 40)
        print("DATASET SELECTION")
        print("-" * 40)
        print("[INFO] Choose how to select your dataset:")
        print("1. Select from available datasets in the data directory")
        print("2. Enter a custom dataset path")
        print("-" * 40)
        while True:
            choice = input("Select option (1-2): ").strip()
            if choice == '1':
                # List available datasets in the data directory
                data_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
                if not data_dirs:
                    print("[ERROR] No datasets found in the specified directory.")
                    print("       Please ensure you have dataset folders available.")
                    return None, None
                print("-" * 40)
                for i, dir_name in enumerate(data_dirs, 1):
                    print(f"  {i}. {dir_name}")
                print("-" * 40)
                try:
                    data_idx = int(input(f"Select dataset (1-{len(data_dirs)}): ")) - 1
                    if data_idx < 0 or data_idx >= len(data_dirs):
                        print(f"[ERROR] Invalid selection. Please enter a number between 1 and {len(data_dirs)}")
                        return None, None
                    selected_test_dir = os.path.join(data_dir, data_dirs[data_idx])
                except ValueError:
                    print("[ERROR] Invalid input. Please enter a number.")
                    return None, None
                print(f"[INFO] Loading data from: {data_dirs[data_idx]}...")
                images, labels = self.load_dataset(selected_test_dir)
                return images, labels
            elif choice == '2':
                while True:
                    custom_path = input("Enter the full path to your dataset directory (or type 'q' to go back): ").strip()
                    if custom_path.lower() == 'q':
                        break
                    custom_path = os.path.join(custom_path, '')
                    if not os.path.isdir(custom_path):
                        print(f"[ERROR] The path '{custom_path}' is not a valid directory. Please try again.")
                        continue
                    print(f"[INFO] Loading data from: {custom_path} ...")
                    images, labels = self.load_dataset(custom_path)
                    return images, labels
            else:
                print("[ERROR] Please enter 1 or 2.")
                continue
    
    def npzCheck(self, data_dir):
        print("\n" + "-" * 40)
        print("DATASET SELECTION")
        print("-" * 40)
        print(f"[INFO] Checking for .npz files in: {data_dir}")
        
        # verify directory exists
        if not os.path.exists(data_dir):
            print(f"[ERROR] Directory {data_dir} does not exist")
            return None, None
        
        # get available test data files with debug info
        files = sorted([file for file in os.listdir(data_dir) if file.endswith('.npz')])
        print(f"[INFO] Found {len(files)} .npz file(s)")
        
        if not files:
            print("[ERROR] No .npz files found in the specified directory.")
            print("       Please ensure you have processed data files available.")
            return None, None
        
        print("-" * 40)
        # print available files with full paths
        for i, filename in enumerate(files, 1):
            full_path = os.path.join(data_dir, filename)
            print(f"  {i}. {filename}")
            print(f"     Path: {full_path}")
        
        print("-" * 40)
        
        # select test file
        try:
            idx = int(input(f"Select dataset (1-{len(files)}): ")) - 1
            if idx < 0 or idx >= len(files):
                print(f"[ERROR] Invalid selection. Please enter a number between 1 and {len(files)}")
                return None, None
                
            selected_file = os.path.join(data_dir, files[idx])
            print(f"[INFO] Loading data from: {selected_file}")
            
            # load the file
            try:
                data = np.load(selected_file)
                if 'images' not in data or 'labels' not in data:
                    print(f"[ERROR] File missing required arrays 'images' or 'labels'")
                    print(f"       Available arrays: {data.files}")
                    return None, None
                    
                images = data['images']
                labels = data['labels']
                
                print("\n" + "-" * 40)
                print("DATASET LOADED SUCCESSFULLY")
                print("-" * 40)
                print(f"[INFO] Images shape: {images.shape}")
                print(f"[INFO] Labels shape: {labels.shape}")
                print(f"[INFO] Total samples: {len(images):,}")
                print(f"[INFO] Unique labels: {len(set(labels))}")
                print("-" * 40)
                
                return images, labels
                
            except Exception as e:
                print(f"[ERROR] Error loading npz file: {e}")
                return None, None
                
        except ValueError:
            print("[ERROR] Invalid input. Please enter a number.")
            return None, None