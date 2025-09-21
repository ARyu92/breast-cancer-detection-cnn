import sys
import os
import numpy as np
import hashlib
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from keras import regularizers
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras import mixed_precision
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf, gc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

mixed_precision.set_global_policy("mixed_float16")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing.data_processor import dataProcessor
from model.breast_cancer_model import BreastCancerModel

#This class is responsible for supervising the model training rounds.
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

        self.model = BreastCancerModel()

        return
    
    #This function takes in the training history details and plots it on a graph.
    def plot_training_history(self, histories):
        plt.figure(figsize=(15, 5))

        #Plots the Training and validation sensitivity
        plt.subplot(1, 3, 1)
        for i, history in enumerate(histories, 1):
            hist = history.history
            epochs = range(1, len(hist['loss']) + 1)
            if 'sensitivity' in hist:
                plt.plot(epochs, hist['sensitivity'], label=f'Run {i} Train Sensitivity')
            if 'val_sensitivity' in hist:
                plt.plot(epochs, hist['val_sensitivity'], label=f'Run {i} Val Sensitivity')
        plt.xlabel('Epochs')
        plt.ylabel('Sensitivity')
        plt.title('Sensitivity')
        plt.legend()
        plt.grid(True)

        #Plots the Training and validation precision
        plt.subplot(1, 3, 2)
        for i, history in enumerate(histories, 1):
            hist = history.history
            epochs = range(1, len(hist['loss']) + 1)
            if 'precision' in hist:
                plt.plot(epochs, hist['precision'], label=f'Run {i} Train Precision')
            if 'val_precision' in hist:
                plt.plot(epochs, hist['val_precision'], label=f'Run {i} Val Precision')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.title('Precision')
        plt.legend()
        plt.grid(True)

        #Plots the Training and validation AUC
        plt.subplot(1, 3, 3)
        for i, history in enumerate(histories, 1):
            hist = history.history
            epochs = range(1, len(hist['loss']) + 1)
            if 'auc' in hist:
                plt.plot(epochs, hist['auc'], label=f'Run {i} Train AUC')
            if 'val_auc' in hist:
                plt.plot(epochs, hist['val_auc'], label=f'Run {i} Val AUC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.title('AUC')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    #This method performs inference of a test set and plots the confusion matrix. 
    def plot_confusion_matrix(self, model, X_test, Y_test, threshold=0.20):
        # Get predictions
        y_pred_probs = model.neural_network.predict(X_test, batch_size=8)
        y_pred = (y_pred_probs >= threshold).astype(int).ravel()

        # True labels
        y_true = Y_test.ravel()

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot and display the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
        disp.plot(cmap=plt.cm.Blues, values_format="d")
        plt.title("Confusion Matrix")
        plt.show()

        return cm

    #This function assigns the tensors into training, validation and test datasets from disk according to paths found in the meta_data.csv file
    def load_training_data(self, meta_data_path, validate = True):
        if validate:
            data_split = [0.70, 0.20, 0.10]
        else:
            data_split = [0.90, 0, 0.10]

        # Retrieve data.
        self.X_train, self.Y_train, self.X_val, self.Y_val,self.X_test,self.Y_test = self.data_processor.prepare_tensors(meta_data_path, split = data_split)

        self.X_train, self.X_val, self.X_test = self.data_processor.z_score_normalization(self.X_train, self.X_val, self.X_test, validate = validate)

        # Force dtype consistency
        self.X_train = np.asarray(self.X_train, dtype="float32")
        self.X_val   = np.asarray(self.X_val, dtype="float32")
        self.X_test  = np.asarray(self.X_test, dtype="float32")

        self.Y_train = np.asarray(self.Y_train, dtype="int32")
        self.Y_val   = np.asarray(self.Y_val, dtype="int32")
        self.Y_test  = np.asarray(self.Y_test, dtype="int32")


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

    #This function performs the entire training pipeline
    def training_pipeline(self, epochs_1 = 32, epochs_2 = 10):
        self.load_training_data("../data/meta_data_A.csv")

        # Phase 1: train head with frozen backbone
        self.model.build_split_network()
        self.model.compile(learning_rate=1e-3)

        history1 = self.model.train(
            self.X_train, self.Y_train,
            self.X_val, self.Y_val,
            epochs=epochs_1, batch_size=8
        )

        # Phase 2: unfreeze last 20 layers and fine-tune
        for backbone in [self.model.cc_backbone, self.model.mlo_backbone]:
            for layer in backbone.layers[-8:]:
                if not isinstance(layer, BatchNormalization):
                    layer.trainable = True

        self.model.compile(learning_rate=1e-5)

        #Clear the memory to avoid OOM issues and memory hole problems.
        tf.keras.backend.clear_session()
        gc.collect()

        history2 = self.model.train(
            self.X_train, self.Y_train,
            self.X_val, self.Y_val,
            epochs= epochs_2, batch_size=4
        )

        #Plot the training history, test the model with unseen test data and plot confusion matrix.
        self.plot_training_history([history1, history2])
        self.plot_confusion_matrix(self.model, self.X_test, self.Y_test)
        #Save the model
        self.model.save_model("training_run")

    #Performs the final model training without a validation set.
    def final_training(self, epochs_1 = 32, epochs_2 = 10):
        self.load_training_data("../data/meta_data_A.csv", validate = False)

        # Phase 1: train head with frozen backbone
        self.model.build_split_network()
        self.model.compile(learning_rate=1e-3)

        history1 = self.model.train(
            self.X_train, self.Y_train,
            self.X_val, self.Y_val,
            epochs=epochs_1, batch_size=8
        )

        # Phase 2: unfreeze last 8 layers and fine-tune
        for layer in self.model.cc_backbone.layers[-8:]:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = True
        self.model.compile(learning_rate=1e-5)

        #Clear the memory to avoid OOM issues and memory hole problems.
        tf.keras.backend.clear_session()
        gc.collect()

        history2 = self.model.train(
            self.X_train, self.Y_train,
            self.X_val, self.Y_val,
            epochs=epochs_2, batch_size=4
        )
        self.plot_training_history([history1, history2])

        self.plot_confusion_matrix(self.model, self.X_test, self.Y_test)

        self.model.save_model("model_test", self.data_processor.training_mean, self.data_processor.training_std)
