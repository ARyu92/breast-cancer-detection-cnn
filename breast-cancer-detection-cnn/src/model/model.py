from tensorflow import keras
from tensorflow.keras import layers
import sys
import os


class Model:
    def __init__(self):
        self.neural_network = None
        return
    
    #Initialize the model
    def build_network(self, layer_input):
        self.neural_network = keras.Sequential()
        self.neural_network.add(layers.Input(shape = (400, 400, 1)))
        #Iterate through the layer inputs and add them.

        for layer in layer_input:
            if layer["layer_type"] == "Conv2D":
                self.neural_network.add(layers.Conv2D(layer["filters"], layer["kernel"], activation = layer["activation"]))
                self.neural_network.add(layers.MaxPooling2D())
            
            if layer ["layer_type"] == "Dense":
                self.neural_network.add(layers.Dense(layer["units"], activation = layer["activation"]))
            
            if layer ["layer_type"] == "Flatten":
                self.neural_network.add(layers.Flatten())

        return self.neural_network
    
    #Compile the neural network model
    def compile(self, learning_rate = 1e-4):
        self.neural_network.compile(
            optimizer = keras.optimizers.Adam(learning_rate),
            loss = "sparse_categorical_crossentropy",
            metrics = ["accuracy"]
        )
    
    def train(self, training_data, training_labels, validation_data, validation_label, epochs = 10, batch_size = 32):
        return self.neural_network.fit(
            x = training_data,
            y = training_labels,
            validation_data = (validation_data, validation_label),
            epochs = epochs,
            batch_size= batch_size
        )