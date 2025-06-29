import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
from model import Model
import sys
import time
import random
import matplotlib.pyplot as plt

def show_loading_throbber(message, duration=None, update_interval=0.1):
    """
    Display a loading throbber with optional duration
    
    Args:
        message (str): The message to display
        duration (float, optional): How long to show the throbber (in seconds)
        update_interval (float): How often to update the throbber (in seconds)
    """
    throbber_chars = ['|', '/', '-', '\\']
    throbber_idx = 0
    start_time = time.time()
    
    try:
        while True:
            # Clear the line and show throbber
            print(f"\r[INFO] {message} {throbber_chars[throbber_idx]}", end='', flush=True)
            
            # Check if duration has elapsed
            if duration and (time.time() - start_time) >= duration:
                break
                
            time.sleep(update_interval)
            throbber_idx = (throbber_idx + 1) % len(throbber_chars)
            
    except KeyboardInterrupt:
        pass
    finally:
        # Clear the throbber line
        print("\r" + " " * (len(message) + 20) + "\r", end='', flush=True)

def show_progress_with_throbber(message, current, total, start_time, update_interval=0.1):
    """
    Show progress with a throbber for operations that don't have natural progress updates
    
    Args:
        message (str): The message to display
        current (int): Current progress
        total (int): Total items
        start_time (float): Start time for ETA calculation
        update_interval (float): How often to update the throbber
    """
    throbber_chars = ['|', '/', '-', '\\']
    throbber_idx = 0
    
    progress = (current / total) * 100 if total > 0 else 0
    elapsed = time.time() - start_time
    
    # Calculate ETA
    if progress > 0:
        avg_time_per_item = elapsed / current
        remaining_items = total - current
        estimated_remaining = avg_time_per_item * remaining_items
        
        if estimated_remaining > 60:
            minutes = int(estimated_remaining // 60)
            seconds = int(estimated_remaining % 60)
            eta_str = f"{minutes}m {seconds}s"
        elif estimated_remaining > 1:
            eta_str = f"{estimated_remaining:.1f}s"
        else:
            eta_str = f"{estimated_remaining*1000:.0f}ms"
        
        print(f"\r[INFO] {message} {throbber_chars[throbber_idx]} - {current:,}/{total:,} ({progress:.1f}%) - ETA: {eta_str}", end='', flush=True)
    else:
        print(f"\r[INFO] {message} {throbber_chars[throbber_idx]} - {current:,}/{total:,} ({progress:.1f}%)", end='', flush=True)
    
    return (throbber_idx + 1) % len(throbber_chars)

def result_helper(model, num, test_images, test_labels, save_csv):
    correct_predictions = 0
    total_predictions = 0
    results = []
    
    # Safety check for test_images
    if test_images is None or len(test_images) == 0:
        print("[ERROR] test_images is None or empty")
        return [], 0, 0
    
    print("\n" + "-" * 40)
    print("TEST CONFIGURATION")
    print("-" * 40)
    rand_choice = get_yes_no("Randomise test case order? (y/n): ")
    
    # Ensure rand_choice is a boolean
    if rand_choice is None:
        rand_choice = False
    
    print(f"[INFO] Randomization: {'Enabled' if rand_choice else 'Disabled'}")
    print(f"[INFO] Test cases to process: {num:,}")
    print("-" * 40)

    for i in range(num):
        try:
            if rand_choice:
                idx = random.randint(0, len(test_images) - 1)
            else:
                idx = i

            result = model.result(test_images, test_labels, idx)
            predicted_label = model.predict(test_images[idx])
            model.plot(test_images[idx], predicted_label)

            if result:
                total_predictions += 1
                if result['correct']:
                    correct_predictions += 1
                
            if save_csv and result:
                results.append(result)

        except Exception as e:
            print(f"[ERROR] Error processing test case {i+1}: {e}")
            continue
    
    # Calculate and display average accuracy
    if total_predictions > 0:
        average_accuracy = (correct_predictions / total_predictions) * 100
        print("\n" + "=" * 50)
        print("TEST RESULTS SUMMARY")
        print("=" * 50)
        print(f"[INFO] Total predictions: {total_predictions:,}")
        print(f"[SUCCESS] Correct predictions: {correct_predictions:,}")
        print(f"[ERROR] Incorrect predictions: {total_predictions - correct_predictions:,}")
        print(f"[INFO] Average accuracy: {average_accuracy:.2f}%")
        print("=" * 50)
    
    return results, correct_predictions, total_predictions

def test_case_num(test_images):
    # Safety check for test_images
    if test_images is None or len(test_images) == 0:
        print("[ERROR] No test images available")
        return 0
    
    print("\n" + "-" * 40)
    print("TEST CASE CONFIGURATION")
    print("-" * 40)
    print(f"[INFO] Available test cases: {len(test_images):,}")
    print("-" * 40)
        
    while True:
        try:
            num = int(input("How many test cases to run?: ").strip())
            if num <= 0:
                print("[ERROR] Please enter a positive number.")
                continue
            if num > len(test_images):
                print(f"[WARNING] You requested {num:,} test cases but only {len(test_images):,} are available.")
                print(f"         Using all {len(test_images):,} available test cases.")
                num = len(test_images)
            break
        except ValueError:
            print("[ERROR] Invalid input. Please enter a number.")
            continue
        except KeyboardInterrupt:
            print("\n[WARNING] Operation cancelled by user.")
            raise
        except EOFError:
            print("\n[WARNING] End of input detected.")
            raise
    
    print(f"[SUCCESS] Configured to run {num:,} test case(s)")
    return num

def save_to_csv(results):
    try:
        print("\n" + "-" * 40)
        print("SAVE RESULTS TO CSV")
        print("-" * 40)
        
        results_dir = os.path.join("content", "results")
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"[INFO] Results will be saved to: {results_dir}")
        print(f"[INFO] Number of results to save: {len(results):,}")
        print("-" * 40)
        
        csv_filename = input("Enter CSV file name (without extension): ").strip()
        if not csv_filename:
            csv_filename = "test_results"
        
        csv_path = os.path.join(results_dir, csv_filename + ".csv")
        
        from dataUtil import save_results_to_csv 
        save_results_to_csv(results, csv_path)
        
        print(f"[SUCCESS] Results saved successfully to: {csv_filename}.csv")
        results.clear()
        
    except Exception as e:
        print(f"[ERROR] Error saving CSV: {e}")
        print("       Please check the file path and permissions.")

def get_valid_int_input(prompt, min_val, max_val):
    while True:
        try:
            choice = input(prompt).strip()
            if not choice:  # Handle empty input
                print(f"[ERROR] Please enter a number between {min_val} and {max_val}")
                continue
                
            choice_int = int(choice)
            if min_val <= choice_int <= max_val:
                return choice_int
            else:
                print(f"[ERROR] Please enter a number between {min_val} and {max_val}")
        except ValueError:
            print(f"[ERROR] Invalid input: '{choice}'. Please enter a number between {min_val} and {max_val}")
        except KeyboardInterrupt:
            print("\n[WARNING] Operation cancelled by user.")
            raise
        except EOFError:
            print("\n[WARNING] End of input detected.")
            raise

def get_yes_no(prompt):
    while True:
        try:
            choice = input(prompt).lower().strip()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            elif choice == '':
                print("[ERROR] Please enter 'y' or 'n'.")
            else:
                print(f"[ERROR] Invalid choice: '{choice}'. Please enter 'y' or 'n'.")
        except (KeyboardInterrupt, EOFError):
            print("\n[WARNING] Operation cancelled by user.")
            raise
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            continue

def select_model(model_dir):
    print("\n" + "-" * 40)
    print("MODEL SELECTION")
    print("-" * 40)
    
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.keras')])
    if not model_files:
        print("[ERROR] No models found in the models directory.")
        print("       Please train a model first using option 1 from the main menu.")
        return None

    print(f"[INFO] Found {len(model_files)} model(s) in: {model_dir}")
    print()
    
    for i, file in enumerate(model_files, 1):
        print(f"  {i}. {file}")
    
    print("-" * 40)
    
    while True:
        try:
            model_choice = input(f"Select a model (1-{len(model_files)}) or 'q' to quit: ").strip()
            if model_choice.lower() == 'q':
                print("[INFO] Returning to main menu...")
                return None
                
            if not model_choice:
                print("[ERROR] Please enter a number or 'q' to quit.")
                continue
                
            model_idx = int(model_choice) - 1
            if model_idx < 0 or model_idx >= len(model_files):
                print(f"[ERROR] Invalid selection. Please enter a number between 1 and {len(model_files)}.")
                continue
                
            selected_model = os.path.join(model_dir, model_files[model_idx])
            print(f"[SUCCESS] Selected: {model_files[model_idx]}")
            return selected_model
            
        except ValueError:
            print(f"[ERROR] Invalid input: '{model_choice}'. Please enter a number or 'q' to quit.")
        except KeyboardInterrupt:
            print("\n[WARNING] Operation cancelled by user.")
            return None
        except EOFError:
            print("\n[WARNING] End of input detected.")
            return None

class Main():
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.dirname(self.base_dir), "content", "data",)
        self.model_dir = os.path.join(os.path.dirname(self.base_dir), "content", "saved_models")
        self._DatasetLoader = None
        self._CharacterImageGenerator = None  # add underscore prefix for private variable (add to log later)

    @property
    def DatasetLoader(self):
        if self._DatasetLoader is None:
            from dataUtil import DatasetLoader
            self._DatasetLoader = DatasetLoader(self.data_dir, self.base_dir)
        return self._DatasetLoader
    
    @property
    def CharacterImageGenerator(self):
        if self._CharacterImageGenerator is None:
            from imgUtil import CharacterImageGenerator
            self._CharacterImageGenerator = CharacterImageGenerator()
        return self._CharacterImageGenerator

    def trainingMode(self, data_dir, model_dir):
        print("\n" + "=" * 50)
        print("TRAINING MODE")
        print("=" * 50)
        while True:
            print("\n" + "-" * 30)
            print("DATASET MENU")
            print("-" * 30)
            print("1. Process dataset")
            print("2. Load processed dataset (.npz)")
            print("3. Return to main menu")
            print("-" * 30)
            choice = get_valid_int_input("Please select an option (1-3): ", 1, 3)
            if choice == 3:
                print("Returning to main menu...")
                return

            if not os.path.exists(data_dir):
                print(f"Training directory not found at {data_dir}. Creating directory...")
                os.makedirs(data_dir, exist_ok=True)

            if choice == 1:
                print("[INFO] Processing dataset. This may take a while...")
                show_loading_throbber("Processing dataset", duration=1.0)
                train_images, train_labels = self.DatasetLoader.dataDirCheck(data_dir)
            elif choice == 2:
                print("[INFO] Loading processed dataset (.npz). This may take a while...")
                show_loading_throbber("Loading processed dataset", duration=1.0)
                train_images, train_labels = self.DatasetLoader.npzCheck(data_dir)

            if train_images is None or train_labels is None:
                print("Failed to load training data. Please check your dataset.")
                continue
            
            # model training section
            try:
                print("\n" + "-" * 40)
                print("MODEL TRAINING")
                print("-" * 40)
                
                print("[INFO] Initializing model...")
                show_loading_throbber("Initializing model", duration=1.0)
                model = Model()
                print("[SUCCESS] Model initialized successfully")
                
                print("[INFO] Starting training process...")
                history = model.train(train_images, train_labels)
                
                if history is None:
                    print("[ERROR] Training failed or was cancelled")
                    continue
                    
                print("[SUCCESS] Training completed successfully")
                
                # plot training history against validation loss
                print("[INFO] Generating training plots...")
                show_loading_throbber("Generating plots", duration=0.5)
                plt.figure(figsize=(10, 5))
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title('Model Loss During Training')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()

                print("[INFO] Saving model...")
                file_name = input("Enter a name for the model file (without extension): ").strip()
                show_loading_throbber("Saving model", duration=0.5)
                model.save_model(os.path.join(model_dir, f"{file_name}.keras"))
                print(f"[SUCCESS] Model saved as {file_name}.keras")
                return

            except Exception as e:
                print(f"[ERROR] Training error: {e}")
                print("Please check the dataset and try again.")
                continue

    def testMode(self, data_dir, model_dir):
        import numpy as np
        print("\n" + "=" * 50)
        print("TEST MODE")
        print("=" * 50)
        while True:
            print("\n" + "-" * 30)
            print("DATASET MENU")
            print("-" * 30)
            print("1. Process dataset")
            print("2. Load processed dataset (.npz)")
            print("3. Return to main menu")
            print("-" * 30)
            choice = get_valid_int_input("Please select an option (1-3): ", 1, 3)
            if choice == 3:
                print("Returning to main menu...")
                return

            if not os.path.exists(data_dir):
                print(f"Test directory not found at {data_dir}. Creating directory...")
                os.makedirs(data_dir, exist_ok=True)

            if choice == 1:
                print("[INFO] Processing dataset.")
                show_loading_throbber("Processing dataset", duration=1.0)
                test_images, test_labels = self.DatasetLoader.dataDirCheck(data_dir)
            elif choice == 2:
                print("[INFO] Loading processed dataset (.npz).")
                show_loading_throbber("Loading processed dataset", duration=1.0)
                test_images, test_labels = self.DatasetLoader.npzCheck(data_dir)

            if test_images is None or test_labels is None or len(test_images) == 0 or len(test_labels) == 0:
                print("Error: No test data available. Please check your dataset.")
                continue
            break

        model_path = select_model(model_dir)
        if not model_path:
            return

        # load model with error handling
        try:
            print("\nLoading model...")
            show_loading_throbber("Loading model", duration=1.0)
            model = Model()
            model.load_model(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please check if the model file is corrupted or try a different model.")
            return

        while True:
            print("\n" + "-" * 30)
            print("TEST MENU")
            print("-" * 30)
            print("1. Visualised test cases")
            print("2. Automate test cases")
            print("3. Return to main menu")
            print("-" * 30)
            try:
                user_input = get_valid_int_input("Please select an option (1-3): ", 1, 3)
                if user_input == 3:
                    print("Returning to main menu...")
                    return
                save_csv = get_yes_no("Save results to CSV? (y/n): ")
                if user_input == 1:
                    num = test_case_num(test_images)
                    if num == 0:
                        continue
                    results, correct_predictions, total_predictions = result_helper(model, num, test_images, test_labels, save_csv)
                    if save_csv and results:
                        print("[INFO] Saving results to CSV...")
                        show_loading_throbber("Saving CSV", duration=0.5)
                        save_to_csv(results)
                    break
                elif user_input == 2:
                    num = test_case_num(test_images)
                    if num == 0:
                        continue
                    
                    print("\n" + "-" * 40)
                    print("AUTOMATED TESTING")
                    print("-" * 40)
                    
                    start = time.time()
                    correct_predictions = 0
                    total_predictions = 0
                    results = []
                    
                    if test_images is None or len(test_images) == 0:
                        print("[ERROR] test_images is None or empty")
                        continue
                    
                    if test_labels is None or len(test_labels) == 0:
                        print("[ERROR] test_labels is None or empty")
                        continue
                    
                    rand_choice = get_yes_no("Randomise test case order? (y/n): ")
                    if rand_choice is None:
                        rand_choice = False
                    
                    print(f"[INFO] Randomization: {'Enabled' if rand_choice else 'Disabled'}")
                    print(f"[INFO] Processing {num:,} test cases...")
                    print("-" * 40)
                    
                    try:
                        # Loading throbber characters with consistent 0.1s interval
                        throbber_chars = ['|', '/', '-', '\\']
                        throbber_idx = 0
                        last_throbber_update = time.time()
                        throbber_interval = 0.1  # Consistent with other throbbers
                        
                        for i in range(num):
                            try:
                                if rand_choice:
                                    idx = random.randint(0, len(test_images) - 1)
                                else:
                                    idx = i
                                
                                result = model.result(test_images, test_labels, idx)
                                if result:
                                    total_predictions += 1
                                    if result['correct']:
                                        correct_predictions += 1
                                
                                if save_csv and result:
                                    results.append(result)
                                    
                                # Progress indicator with throbber and time estimation
                                if (i + 1) % 100 == 0 or (i + 1) == num:
                                    # Clear any existing throbber line first
                                    print("\r" + " " * 50 + "\r", end='', flush=True)
                                    
                                    progress = ((i + 1) / num) * 100
                                    elapsed = time.time() - start
                                    
                                    # Calculate estimated time remaining
                                    if progress > 0:
                                        avg_time_per_test = elapsed / (i + 1)
                                        remaining_tests = num - (i + 1)
                                        estimated_remaining = avg_time_per_test * remaining_tests
                                        
                                        if estimated_remaining > 60:
                                            minutes = int(estimated_remaining // 60)
                                            seconds = int(estimated_remaining % 60)
                                            eta_str = f"{minutes}m {seconds}s"
                                        elif estimated_remaining > 1:
                                            eta_str = f"{estimated_remaining:.1f} seconds"
                                        else:
                                            eta_str = f"{estimated_remaining*1000:.0f} milliseconds"
                                        
                                        print(f"[INFO] Progress: {i + 1:,}/{num:,} ({progress:.1f}%) - ETA: {eta_str}")
                                    else:
                                        print(f"[INFO] Progress: {i + 1:,}/{num:,} ({progress:.1f}%)")
                                else:
                                    # Show throbber with consistent 0.1s interval
                                    current_time = time.time()
                                    if current_time - last_throbber_update >= throbber_interval:
                                        print(f"\r[INFO] Processing... {throbber_chars[throbber_idx]}", end='', flush=True)
                                        throbber_idx = (throbber_idx + 1) % len(throbber_chars)
                                        last_throbber_update = current_time
                                    
                            except Exception as e:
                                print(f"[ERROR] Error processing test case {i+1}: {e}")
                                continue
                        
                        # Clear the throbber line
                        print("\r" + " " * 50 + "\r", end='', flush=True)
                        
                        end = time.time()
                        elapsed_time = round(end - start, 2)
                        print(f"[INFO] Elapsed time: {elapsed_time}s")
                        
                        if total_predictions > 0:
                            average_accuracy = (correct_predictions / total_predictions) * 100
                            print("\n" + "=" * 50)
                            print("AUTOMATED TEST RESULTS")
                            print("=" * 50)
                            print(f"[INFO] Total predictions: {total_predictions:,}")
                            print(f"[SUCCESS] Correct predictions: {correct_predictions:,}")
                            print(f"[ERROR] Incorrect predictions: {total_predictions - correct_predictions:,}")
                            print(f"[INFO] Average accuracy: {average_accuracy:.2f}%")
                            print(f"[INFO] Processing time: {elapsed_time}s")
                            print("=" * 50)
                        
                        if save_csv and results:
                            print("[INFO] Saving results to CSV...")
                            show_loading_throbber("Saving CSV", duration=0.5)
                            save_to_csv(results)
                            
                    except Exception as e:
                        print(f"[ERROR] Error during testing: {e}")
                    
                    break
            except ValueError:
                print("[ERROR] Invalid input. Please enter a number between 1 and 3.")
                continue
            except KeyboardInterrupt:
                print("\n[WARNING] Operation cancelled by user.")
                return
            except Exception as e:
                print(f"[ERROR] Unexpected error: {e}")
                print("Please try again or select option 3 to return to main menu.")
                continue

main = Main()
if __name__ == "__main__":
    print("=" * 50)
    print("Welcome to Max's TensorFlow OCR!")
    print("=" * 50)
    
    while True:
        try:
            print("-" * 30)
            print("MAIN MENU")
            print("-" * 30)
            print("1. Train a new model")
            print("2. Test an existing model")
            print("3. Generate custom dataset of character images")
            print("4. Exit")
            print("-" * 30)

            choice = input("Please select an option (1-4): ").strip()

            if choice == '1':
                print("\nStarting Training Mode...")
                main.trainingMode(main.data_dir, main.model_dir)

            elif choice == '2':
                print("\nStarting Test Mode...")
                main.testMode(main.data_dir, main.model_dir)

            elif choice == '3':
                print("\nStarting Dataset Generation...")
                print("\n" + "-" * 40)
                print("IMAGE GENERATION CONFIGURATION")
                print("-" * 40)
                print("[INFO] Random rotation adds ±20 degrees variation to each character")
                print("[INFO] This helps the model learn to recognize rotated text")
                print("-" * 40)
                apply_rotation = get_yes_no("Apply random rotation to images? (y/n): ")
                if apply_rotation is None:
                    apply_rotation = True  # Default to True if user cancels
                print(f"[INFO] Random rotation: {'Enabled' if apply_rotation else 'Disabled'}")
                print("-" * 40)
                main.CharacterImageGenerator.generateImages(apply_rotation)

            elif choice == '4':
                print("\nThank you for using Max's TensorFlow OCR! (^_^)")
                print("Exiting program...")
                sys.exit(0)

            else:
                print(f"\n[ERROR] Invalid choice: '{choice}'")
                print("Please enter a number between 1 and 4.")

        except KeyboardInterrupt:
            print("\n\n[WARNING] Operation cancelled by user.")
            print("Exiting program...")
            sys.exit(0)
        except EOFError:
            print("\n\n[WARNING] End of input detected.")
            print("Exiting program...")
            sys.exit(0)
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}")
            print("Please try again or contact support if the problem persists.")
            continue