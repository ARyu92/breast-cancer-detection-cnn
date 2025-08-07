from tensorflow import keras
from tensorflow.keras import layers
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

class Model:
    def __init__(self):
        self.neural_network = None
        return
    
    #Initialize the model
    def build_network(self, layer_input):
        self.neural_network = keras.Sequential()
        self.neural_network.add(layers.Input(shape=(640, 640, 2)))

        for layer in layer_input:
            if layer["layer_type"] == "Conv2D":
                self.neural_network.add(layers.Conv2D(
                    filters=layer["filters"],
                    kernel_size=layer["kernel"],
                    strides=layer.get("stride", 1),
                    activation=layer["activation"],
                    padding="same"
                ))

            elif layer["layer_type"] == "MaxPool":
                self.neural_network.add(layers.MaxPooling2D(
                    pool_size=layer["pool_size"]
                ))

            elif layer["layer_type"] == "Dense":
                self.neural_network.add(layers.Dense(
                    units=layer["units"],
                    activation=layer["activation"],
                    dtype=layer.get("dtype", None)
                ))

            elif layer["layer_type"] == "Dropout":
                self.neural_network.add(layers.Dropout(
                    rate=layer["rate"]
                ))

            elif layer["layer_type"] == "BatchNorm":
                self.neural_network.add(layers.BatchNormalization())

            elif layer["layer_type"] == "Flatten":
                self.neural_network.add(layers.Flatten())

            elif layer["layer_type"] == "GlobalAvgPool":
                self.neural_network.add(layers.GlobalAveragePooling2D())

        return self.neural_network
    
    #Compile the neural network model
    def compile(self, learning_rate = 1e-4):
        self.neural_network.compile(
            optimizer = keras.optimizers.Adam(learning_rate),
            loss = "binary_crossentropy",
            metrics = ["accuracy"]
        )
    
    def train(self, training_data, training_labels, validation_data, validation_label, epochs = 5, batch_size = 2):
        return self.neural_network.fit(
            x = training_data,
            y = training_labels,
            validation_data = (validation_data, validation_label),
            epochs = epochs,
            batch_size= batch_size
        )
    
    def evaluate(self, data, labels):
        loss, accuracy = self.neural_network.evaluate(data, labels, verbose=2, batch_size = 1)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        return
    
    #This function saves the model to a given path.
    def save_model(self, path, model_name):
        base_path = path + model_name
        model_path = base_path
        name = model_name
        counter = 1

        while os.path.exists(model_path):
            name = f"{model_name}_{counter}"
            model_path = os.path.join(path, name)
            counter += 1

        os.makedirs(model_path)
        self.neural_network.save(f"{model_path}/{name}.keras")
        return
    
    #This function loads the model at a given path
    def load_model(self, path):
        return keras.models.load_model(path)
