from loadData import DatasetLoader
from model import Model
import matplotlib.pyplot as plt
import numpy as np
import os

#funny name for my ai :)
# Franklin is a name that is often associated with intelligence and wisdom, making it a fitting choice for an AI.
# The name Franklin is also a nod to Benjamin Franklin, a polymath known for his contributions to science, politics, and philosophy.
# This name could suggest that the AI is designed to be versatile and knowledgeable, capable of handling a wide range of tasks and challenges.
def Franklin():
    # Use absolute path relative to the script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(os.path.dirname(current_dir), "content", "data", "training_data")
    test_dir = os.path.join(os.path.dirname(current_dir), "content", "data", "testing_data")

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

    # Load and evaluate on test data
    if os.path.exists(test_dir):
        print("Loading test data...")
        test_loader = DatasetLoader(test_dir)
        test_images, test_labels = test_loader.load_dataset()
        
        # Evaluate model
        print("Evaluating model...")
        test_accuracy = model.accuracy(test_images, test_labels)
        print(f"Test accuracy: {test_accuracy:.2%}")
        
        # Plot some predictions
        for i in range(10):
            idx = np.random.randint(0, len(test_images))
            model.plot_prediction(test_images, idx)

if __name__ == "__main__":
    Franklin()