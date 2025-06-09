from model import Model
import os
import sys

class Main():
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.dirname(self.base_dir), "content", "data",)
        self.model_dir = os.path.join(os.path.dirname(self.base_dir), "content", "saved_models")
        self._DatasetLoader = None
        self._CharacterImageGenerator = None  # Add underscore prefix for private variable

    @property
    def DatasetLoader(self):
        if self._DatasetLoader is None:
            from dataUtil import DatasetLoader
            self._DatasetLoader = DatasetLoader(self.data_dir, self.base_dir)
        return self._DatasetLoader
    
    @property
    def CharacterImageGenerator(self):
        if self._CharacterImageGenerator is None:
            #remeber error: reached maximum recursion depth, 
            #fixed by adding underscore prefix for private variable
            from character_image_generator import CharacterImageGenerator
            self._CharacterImageGenerator = CharacterImageGenerator()
        return self._CharacterImageGenerator

    def trainingMode(self, data_dir, model_dir):
        # Import here instead of at top
        import matplotlib.pyplot as plt
        # Initialize the dataset loader with error checking
        train_images, train_labels = self.DatasetLoader.dataDirCheck(data_dir)
        if train_images is None or train_labels is None:
            return
        
        if not os.path.exists(data_dir):
            print(f"Error: Training directory not found at {data_dir}")
            return
        
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
        model.save_model(os.path.join(model_dir, f"{file_name}.keras"))
        print(f"Model saved as {file_name}.keras")

    def testMode(self, data_dir, model_dir):
        # Import here instead of at top
        import numpy as np
        # Get test data
        test_images, test_labels = self.DatasetLoader.dataDirCheck(data_dir)
        if test_images is None or test_labels is None:
            return
        
        # Get model file paths
        print("Available models:")
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
        model.accuracy(test_images, test_labels)

        if len(test_images) <= 10:
            for i in range(len(test_images)):
                model.plot_prediction(test_images, i)

        else:
            for i in range(10):
                idx = np.random.randint(0, len(test_images))
                model.plot_prediction(test_images, idx)

main = Main()
if __name__ == "__main__":
    while True:
        print("Welcome to Max's TensorFlow OCR!")
        print("1. Train a new model")
        print("2. Test an existing model")
        print("3. Generate custom dataset of character images")
        print("4. Exit")

        choice = input("Please select a valid option (1-3): ")

        if choice == '1':
            main.trainingMode(main.data_dir, main.model_dir)
            print("Returning to main menu...")

        elif choice == '2':
            main.testMode(main.data_dir, main.model_dir)
            print("Returning to main menu...")

        elif choice == '3':
            main.CharacterImageGenerator.process_fonts()

        elif choice == '4':
            print("Exiting program...")
            sys.exit(0) #terminate the program

        else:
            print("Invalid choice. Please select a number between 1 and 3.")
            continue