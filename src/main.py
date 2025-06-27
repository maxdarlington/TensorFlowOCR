import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
from model import Model
import sys
import time
import random

def save_to_csv(results):
    try:
        results_dir = os.path.join("content", "results")
        os.makedirs(results_dir, exist_ok=True)
        csv_filename = input("Enter CSV file name to save results without extension (e.g., results): ").strip()
        if not csv_filename:
            csv_filename = "test_results"
        csv_path = os.path.join(results_dir, csv_filename + ".csv")
        from dataUtil import save_results_to_csv 
        save_results_to_csv(results, csv_path)
        results.clear()
    except Exception as e:
        print(f"Error saving CSV: {e}")

def get_valid_int_input(prompt, min, max):
    while True:
        try:
            choice = int(input(prompt))
            if min <= choice <= max:
                return choice
            else:
                print(f"Please select a number between {min} and {max}")
        except ValueError:
            print("Invalid input. Please enter a numerical character")

def get_yes_no(prompt):
    while True:
        try:
            choice = input(prompt).lower().strip()
            if choice in ['y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            else:
                print("Invalid choice. (y/n)")
                continue
        except ValueError:
            print("Invalid choice. (y/n)")

def select_model(model_dir):
    print("Available models:")
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.keras')])
    if not model_files:
        print("No models found. Please train a model first.")
        return None

    for i, file in enumerate(model_files):
        print(f"{i+1}. {file}")

    while True:
        try:
            model_choice = input(f"Please select a model (1-{len(model_files)}) or 'q' to quit: ").strip()
            if model_choice.lower() == 'q':
                print("Returning to main menu...")
                return None
            model_idx = int(model_choice) - 1
            if model_idx < 0 or model_idx >= len(model_files):
                print(f"Invalid selection. Please enter a number between 1 and {len(model_files)}.")
                continue
            return os.path.join(model_dir, model_files[model_idx])
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")

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
        import matplotlib.pyplot as plt

        # dataset submenu
        print("1. Process dataset")
        print("2. Load processed dataset (.npz)")

        if not os.path.exists(data_dir):
            print(f"Training directory not found at {data_dir}. Creating directory...")
            os.makedirs(data_dir, exist_ok=True)

        choice = get_valid_int_input("Please select a valid option (1-2): ", 1, 2)

        if choice == 1:
            train_images, train_labels = self.DatasetLoader.dataDirCheck(data_dir)
        elif choice == 2:
            train_images, train_labels = self.DatasetLoader.npzCheck(data_dir)

        if train_images is None or train_labels is None:
            print("Failed to load training data. Please check your dataset.")
            return
        
        # model training section
        try:
            print("Initializing model...")
            model = Model()
            history = model.train(train_images, train_labels)
            print("Training model...")
            
            # plot training history against validation loss
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss During Training')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

            file_name = input("Enter a name for the model file (without extension): ")
            model.save_model(os.path.join(model_dir, f"{file_name}.keras"))
            print(f"Model saved as {file_name}.keras")

        except Exception as e:
            print(f"Error: {e}")
            print("Please check the dataset and try again.")
            return

    def testMode(self, data_dir, model_dir):
        import numpy as np
        
        # dataset submenu
        print("1. Process dataset")
        print("2. Load processed dataset (.npz)")
        print("3. Return to main menu")

        test_images, test_labels = self.DatasetLoader.select_dataset(data_dir)

        model_path = select_model(model_dir)
        if not model_path:
            return

        # load model with error handling
        try:
            print("Loading model...")
            model = Model()
            model.load_model(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please check if the model file is corrupted or try a different model.")
            return

        print("1. Visualise test cases")
        print("2. Automate test cases")
        print("3. Return to main menu")

        while True:
            try:
                user_input = int(input("Please select a valid option (1-3): "))
                
                if user_input not in [1, 2, 3]:
                    print("Invalid option. Please select 1, 2, or 3.")
                    continue
                
                if user_input == 3:
                    print("Returning to main menu...")
                    return
                
                # Handle CSV save option
                save_csv = get_yes_no("Save results to CSV? (y/n): ")
                results = []

                if user_input == 1:
                    while True:
                        try:
                            num = int(input("How many test cases to visualise?: "))
                            if num <= 0:
                                print("Please enter a positive number.")
                                continue
                            if num > len(test_images):
                                print(f"Warning: You requested {num} test cases but only {len(test_images)} are available.")
                                num = len(test_images)
                            break
                        except ValueError:
                            print("Invalid input. Please enter a number.")
                            continue
                    
                    correct_predictions = 0
                    total_predictions = 0
                    
                    for i in range(num):
                        try:
                            randidx = random.randint(0, len(test_images) - 1)
                            result = model.result(test_images, test_labels, randidx)
                            predicted_label = model.predict(test_images[randidx])
                            model.plot(test_images[randidx], predicted_label)

                            if result:
                                total_predictions += 1
                                if result['correct']:
                                    correct_predictions += 1
                                
                            if save_csv and result:
                                results.append(result)
                        except Exception as e:
                            print(f"Error processing test case {i+1}: {e}")
                            continue
                    
                    # Calculate and display average accuracy
                    if total_predictions > 0:
                        average_accuracy = (correct_predictions / total_predictions) * 100
                        print(f"\n=== TEST RESULTS ===")
                        print(f"Total predictions: {total_predictions}")
                        print(f"Correct predictions: {correct_predictions}")
                        print(f"Average accuracy: {average_accuracy:.2f}%")
                        print("===================")
                            
                    if save_csv and results:
                        try:
                            csv_path = os.path.join("content", "results")
                            os.makedirs(csv_path, exist_ok=True)
                            from dataUtil import save_results_to_csv
                            save_results_to_csv(results, csv_path)
                            results.clear()
                        except Exception as e:
                            print(f"Error saving CSV: {e}")
                    break

                elif user_input == 2:
                    print("Testing model...")
                    start = time.time()
                    correct_predictions = 0
                    total_predictions = 0
                    
                    try:
                        for i in range(len(test_images)):
                            result = model.result(test_images, test_labels, i)
                            if result:
                                total_predictions += 1
                                if result['correct']:
                                    correct_predictions += 1
                                    
                            if save_csv and result:
                                results.append(result)
                        end = time.time()
                        print(f"Elapsed time: {round(end - start, 2)}s")
                        
                        # Calculate and display average accuracy
                        if total_predictions > 0:
                            average_accuracy = (correct_predictions / total_predictions) * 100
                            print(f"\n=== TEST RESULTS ===")
                            print(f"Total predictions: {total_predictions}")
                            print(f"Correct predictions: {correct_predictions}")
                            print(f"Average accuracy: {average_accuracy:.2f}%")
                            print("===================")
                                
                        #add csv saving

                        break
                    except Exception as e:
                        print(f"Error during testing: {e}")
                        break
                
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 3.")
                continue
            except KeyboardInterrupt:
                print("\nOperation cancelled by user.")
                return
            except Exception as e:
                print(f"Unexpected error: {e}")
                print("Please try again or select option 3 to return to main menu.")
                continue

main = Main()
if __name__ == "__main__":
    while True:
        # main menu loop
        print("Welcome to Max's TensorFlow OCR!")
        print("1. Train a new model")
        print("2. Test an existing model")
        print("3. Generate custom dataset of character images")
        print("4. Exit")

        choice = input("Please select a valid option (1-4): ")

        if choice == '1':
            main.trainingMode(main.data_dir, main.model_dir)

        elif choice == '2':
            main.testMode(main.data_dir, main.model_dir)

        elif choice == '3':
            main.CharacterImageGenerator.generateImages()

        elif choice == '4':
            print("Exiting program...")
            sys.exit(0) #terminate the program

        else:
            print("Invalid choice. Please select a number between 1 and 3.")
            continue