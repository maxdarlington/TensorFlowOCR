from dataUtil import DatasetLoader
from model import Model
import matplotlib.pyplot as plt
import numpy as np
import os

class Main():
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.dirname(self.base_dir), "content", "data",)
        self.save_dir = os.path.join(os.path.dirname(self.base_dir), "content", "saved_models")

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
        data_idx = int(input("Select dataset number: ")) - 1
        if data_idx < 0 or data_idx >= len(data_dirs):
            print("Invalid selection. Exiting test mode.")
            return None, None
        selected_test_dir = os.path.join(data_dir, data_dirs[data_idx])
        
        print(f"Loading data from {data_dirs[data_idx]}...")
        data_loader = DatasetLoader(selected_test_dir)
        images, labels = data_loader.load_dataset()
        return images, labels

    def trainingMode(self, data_dir, save_dir):
        # Initialize the dataset loader with error checking
        training_images, training_labels = self.dataDirCheck(data_dir)
        if training_images is None or training_labels is None:
            return
        
        if not os.path.exists(data_dir):
            print(f"Error: Training directory not found at {data_dir}")
            return
            
        # Load training data
        print("Loading training data...")
        training_loader = DatasetLoader(data_dir)
        train_images, train_labels = training_loader.load_dataset()

        # Initialize and train the model
        print("Initializing model...")
        model = Model()
        history = model.train(train_images, train_labels)
        print("Training model...")
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        file_name = input("Enter a name for the model file (without extension): ")
        model.save_model(os.path.join(save_dir, f"{file_name}.keras"))
        print(f"Model saved as {file_name}.keras")

    def testMode(self, data_dir, model_dir):
        # Get test data
        test_images, test_labels = self.dataDirCheck(data_dir)
        if test_images is None or test_labels is None:
            return
        
        # Get model file paths
        print("\nAvailable models:")
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
        if not model_files:
            print("No models found. Exiting test mode.")
            return
            
        for i, file in enumerate(model_files):
            print(f"{i+1}. {file}")
        
        model_idx = int(input("Select model number: ")) - 1
        if model_idx < 0 or model_idx >= len(model_files):
            print("Invalid selection. Exiting test mode.")
            return
        model_path = os.path.join(model_dir, model_files[model_idx])
        
        # Load the model
        print("Loading model...")
        model = Model()
        model.load_model(model_path)
        
        # Evaluate model
        print("Evaluating model...")
        test_accuracy = model.accuracy(test_images, test_labels)
        print(f"Test Accuracy: {test_accuracy:.2%}")
        
        if len(test_images) <= 10:
            for i in range(len(test_images)):
                model.plot_prediction(test_images, i)
        else:
            for i in range(10):
                idx = np.random.randint(0, len(test_images))
                model.plot_prediction(test_images, idx)

main = Main()
current_dir = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    while True:
        print("Welcome to Max's TensorFlow OCR!")
        print("1. Train a new model")
        print("2. Test an existing model")
        print("3. Exit")
        choice = input("Please select a valid option (1-3): ")

        if choice == '1':
            main.trainingMode(main.data_dir, main.save_dir)
            print("Returning to main menu...")
        elif choice == '2':
            main.testMode(main.data_dir, main.save_dir)
            print("Returning to main menu...")
        elif choice == '3':
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please select a number between 1 and 3.")
            continue