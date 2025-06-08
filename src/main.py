from model import Model
import os
import sys

class Main():
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(os.path.dirname(self.base_dir), "content", "data",)
        self.model_dir = os.path.join(os.path.dirname(self.base_dir), "content", "models")
        self._datasetloader = None

    @property
    def datasetloader(self):
        if self._datasetloader is None:
            from dataUtil import DatasetLoader
            self._datasetloader = DatasetLoader(self.data_dir, self.base_dir)
        return self._datasetloader

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
        images, labels = self.datasetloader.load_dataset(selected_test_dir)
        return images, labels

    def trainingMode(self, data_dir, model_dir):
        # Import here instead of at top
        import matplotlib.pyplot as plt
        # Initialize the dataset loader with error checking
        train_images, train_labels = self.dataDirCheck(data_dir)
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
        test_images, test_labels = self.dataDirCheck(data_dir)
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
        print("3. Exit")
        choice = input("Please select a valid option (1-3): ")

        if choice == '1':
            main.trainingMode(main.data_dir, main.model_dir)
            print("Returning to main menu...")
        elif choice == '2':
            main.testMode(main.data_dir, main.model_dir)
            print("Returning to main menu...")
        elif choice == '3':
            print("Exiting program...")
            sys.exit(0) #terminate the program
        elif choice == '4':
            # List available .npz files
            print("Available processed datasets:")
            npz_files = [f for f in os.listdir(main.data_dir) if f.endswith('.npz')]
            if not npz_files:
                print("No processed datasets found.")
                continue
                
            for i, file in enumerate(npz_files):
                print(f"{i+1}. {file}")
            
            try:
                file_idx = int(input("Select dataset number: ")) - 1
                if file_idx < 0 or file_idx >= len(npz_files):
                    print("Invalid selection.")
                    continue
                
                npz_path = os.path.join(main.data_dir, npz_files[file_idx])
                data = main.datasetloader.load_processed_data(npz_path)
                if data:
                    print(f"Successfully loaded data:")
                    print(f"Images shape: {data['images_shape']}")
                    print(f"Labels shape: {data['labels_shape']}")
            except ValueError:
                print("Invalid input. Please enter a number.")
        else:
            print("Invalid choice. Please select a number between 1 and 3.")
            continue