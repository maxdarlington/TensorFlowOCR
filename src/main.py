from dataUtil import DatasetLoader
from model import Model
import matplotlib.pyplot as plt
import numpy as np
import os

class Main():
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.train_dir = os.path.join(os.path.dirname(self.base_dir), "content", "data", "training_data")
        self.test_dir = os.path.join(os.path.dirname(self.base_dir), "content", "data", "testing_data")
        self.save_dir = os.path.join(os.path.dirname(self.base_dir), "content", "saved_models")

    def trainingMode(self, train_dir, save_dir):
        # Initialize the dataset loader with error checking
        if not os.path.exists(train_dir):
            print(f"Error: Training directory not found at {train_dir}")
            return
            
        # Load training data
        print("Loading training data...")
        training_loader = DatasetLoader(train_dir)
        train_images, train_labels = training_loader.load_dataset()

        # Initialize and train the model
        print("Initializing model...")
        model = Model()
        print("Training model...")
        history = model.train(train_images, train_labels)

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

    def testMode(self, test_dir, model_dir):
    # Load and evaluate on test data
        if os.path.exists(test_dir):
            print("Loading test data...")
            test_loader = DatasetLoader(test_dir)
            test_images, test_labels = test_loader.load_dataset()
            
            # Get model file paths
            print("Available models:")
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
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
            print(f"Test accuracy: {test_accuracy:.2%}")
            
            # Plot some predictions
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
        choice = input("Please select an option (1-3): ")

        if choice == '1':
            main.trainingMode(main.train_dir, main.save_dir)
            continue_choice = input("Do you want to test the model now? (y/n): ")
            if continue_choice.lower() == 'y':
                main.testMode(main.test_dir, main.save_dir)
            else:
                print("Returning to main menu...")
        elif choice == '2':
            main.testMode(main.test_dir, main.save_dir)
            print("Returning to main menu...")
        elif choice == '3':
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please select a number between 1 and 3.")
            continue
