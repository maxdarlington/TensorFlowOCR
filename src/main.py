import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
from model import Model
import sys
import time
import random
from dataUtil import save_results_to_csv

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

        try:
            user_input = input("Please select a valid option (1-2): ")
            choice = int(user_input)
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 2.")
            return

        if choice == 1:     
            train_images, train_labels = self.DatasetLoader.dataDirCheck(data_dir)
            if train_images is None or train_labels is None:
                return
            
            if not os.path.exists(data_dir):
                print(f"Error: Training directory not found at {data_dir}")
                return
            
        elif choice == 2:
            train_images, train_labels = self.DatasetLoader.npzCheck(data_dir)
            if train_images is None or train_labels is None:
                return
            
        else:
            print("Invalid choice. Please select a number between 1 and 2.")
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
        choice = int(input("Please select a valid option (1-2): "))

        if choice == 1:     
            test_images, test_labels = self.DatasetLoader.dataDirCheck(data_dir)
            if test_images is None or test_labels is None:
                return
            
            if not os.path.exists(data_dir):
                print(f"Error: Training directory not found at {data_dir}")
                return
            
        elif choice == 2:
            test_images, test_labels = self.DatasetLoader.npzCheck(data_dir)

        else:
            print("Invalid choice. Please select a number between 1 and 2.")
            return
        
        # get model file paths
        print("Available models:")
        model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.keras')])
        if not model_files:
            print("No models found. Exiting test mode.")
            return
            
        for i, file in enumerate(model_files):
            print(f"{i+1}. {file}")
        
        try:
            model_choice = input(f"Please select a model (1-{len(model_files)}): ")
            model_idx = int(model_choice) - 1
            
            if model_idx < 0 or model_idx >= len(model_files):
                print("Invalid selection. Exiting test mode.")            
                return
                
        except Exception as e:
            print(f"Error: {e}")
            return
            
        model_path = os.path.join(model_dir, model_files[model_idx])
        
        # load model
        print("Loading model...")
        model = Model()
        model.load_model(model_path)

        print("1. Visualise test cases")
        print("2. Automate test cases")

        while True:
            user_input = int(input("Please select a valid option (1-2)"))
            save_csv = input("Save results to CSV? (y/n): ").strip().lower() == 'y'
            results = []

            try:
                if user_input == 1:
                    num = int(input("How many test cases to visualise?: "))
                    for i in range(num):
                        randidx = random.randint(0, len(test_images) - 1)
                        result = model.result(test_images, test_labels, randidx)
                        predicted_label = model.predict(test_images[randidx])
                        model.plot(test_images[randidx], predicted_label)

                        if save_csv and result:
                            results.append(result)
                            
                    if save_csv and results:
                        csv_path = os.path.join("content", "results")
                        save_results_to_csv(results, csv_path)
                        results.clear()
                        break

                elif user_input == 2:
                    for i in range(20):
                        result = model.result(test_images, test_labels, i)
                        if save_csv and result:
                            results.append(result)
                            
                    if save_csv and results:
                        results_dir = os.path.join("content", "data", "results")
                        os.makedirs(results_dir, exist_ok=True)
                        csv_path = os.path.join(results_dir, input("Enter CSV file name to save results (e.g., results.csv): ").strip())
                        save_results_to_csv(results, csv_path)
                        results.clear()
                        break
                
                else:
                    print("Invalid option. Please select 1 or 2.")
                    
            except Exception as e:
                print(f"Error: {e}")
                break

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
            print("Returning to main menu...")
            time.sleep(3)

        elif choice == '2':
            main.testMode(main.data_dir, main.model_dir)
            print("Returning to main menu...")
            time.sleep(3)

        elif choice == '3':
            main.CharacterImageGenerator.generateImages()
            print("Returning to main menu...")
            time.sleep(3)

        elif choice == '4':
            print("Exiting program...")
            sys.exit(0) #terminate the program

        else:
            print("Invalid choice. Please select a number between 1 and 3.")
            time.sleep(3)
            continue