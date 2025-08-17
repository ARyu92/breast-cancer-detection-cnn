import sys
import os
import numpy as np 
import hashlib
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from keras import regularizers
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing.data_processor import dataProcessor
from model.model import Model

class Trainer:
    def __init__(self):
        self.data_processor = dataProcessor()
        #Initialize input data, labels, image meta data.
        self.X = None
        self.Y = None
        
        self.X_train = None
        self.Y_train = None

        self.X_val = None
        self.Y_val = None

        self.X_test = None
        self.Y_test = None

        self.model = Model()

        return
    
    #This function takes in the training history details and plots it on a graph.
    def plot_training_history(self, history):
        # Extract the history dictionary
        hist = history.history
        epochs = range(1, len(hist['loss']) + 1)

        # Plot accuracy
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        if 'accuracy' in hist:
            plt.plot(epochs, hist['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in hist:
            plt.plot(epochs, hist['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training & Validation Accuracy')
        plt.legend()
        plt.grid(True)

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, hist['loss'], label='Training Loss')
        if 'val_loss' in hist:
            plt.plot(epochs, hist['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    #This function performs the entire training pipeline
    def training_pipeline(self):
        # Retrieve data.
        
        self.X_train, self.Y_train, self.X_val, self.Y_val,self.X_test,self.Y_test = self.data_processor.prepare_tensors("../data/meta_data.csv")

        #Last Z-score normalization
        mean = self.X_train.mean(axis=(0, 1, 2), keepdims=True).astype(np.float32)
        std  = (self.X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-6).astype(np.float32)

        # Apply z-score normalization to all sets
        self.X_train = (self.X_train - mean) / std
        self.X_val   = (self.X_val   - mean) / std
        self.X_test  = (self.X_test  - mean) / std

        print("Training Set:")
        print(f"  X_train: {np.array(self.X_train).shape}")
        print(f"  Y_train: {np.array(self.Y_train).shape}")

        print("Validation Set:")
        print(f"  X_val: {np.array(self.X_val).shape}")
        print(f"  Y_val: {np.array(self.Y_val).shape}")

        print("Test Set:")
        print(f"  X_test: {np.array(self.X_test).shape}")
        print(f"  Y_test: {np.array(self.Y_test).shape}")

        print("Train:", np.bincount(self.Y_train))
        print("Val:  ", np.bincount(self.Y_val))


        wd = regularizers.l2(5e-4)

        
        layer_input = [
                    # Block 1
                    {"layer_type": "Conv2D", "filters": 32, "kernel": (3, 3), "activation": "relu"},
                    {"layer_type": "BatchNorm"},
                    {"layer_type": "Conv2D", "filters": 32, "kernel": (3, 3), "activation": "relu"},
                    {"layer_type": "BatchNorm"},
                    {"layer_type": "MaxPool", "pool_size": (2, 2)},  # 400 → 200
                    # Block 2
                    {"layer_type": "Conv2D", "filters": 64, "kernel": (3, 3), "activation": "relu"},
                    {"layer_type": "BatchNorm"},
                    {"layer_type": "Conv2D", "filters": 64, "kernel": (3, 3), "activation": "relu", "stride": 2},  # 200 → 100
                    {"layer_type": "BatchNorm"},
                    # Block 3
                    {"layer_type": "Conv2D", "filters": 128, "kernel": (3, 3), "activation": "relu"},
                    {"layer_type": "BatchNorm"},
                    {"layer_type": "Conv2D", "filters": 128, "kernel": (3, 3), "activation": "relu", "stride": 2},  # 100 → 50
                    {"layer_type": "BatchNorm"},
                    # Output Head
                    {"layer_type": "GlobalAvgPool"},
                    {"layer_type": "Dense", "units": 128, "activation": "relu"},
                    {"layer_type": "Dropout", "rate": 0.3},
                    {"layer_type": "Dense", "units": 1, "activation": "sigmoid", "dtype": "float32"}
        ]



        self.model.build_network(layer_input)
        self.model.compile()

        history = self.model.train(self.X_train, self.Y_train, self.X_val, self.Y_val, epochs = 300)
        self.plot_training_history(history)
        
        self.model.evaluate(self.X_test, self.Y_test)
        self.model.save_model("model_test")

