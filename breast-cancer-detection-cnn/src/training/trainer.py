import sys
import os
import numpy as np 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing.data_processor import dataProcessor
from model.model import Model

class Trainer:
    def __init__(self):
        self.data_processor = dataProcessor()
        #Initialize input data, labels, image meta data.
        self.X = None
        self.Y = None
        self.Z = None

        self.X_train = None
        self.Y_train = None
        self.Z_train = None

        self.X_val = None
        self.Y_val = None
        self.Z_val = None

        self.X_test = None
        self.Y_test = None
        self.Z_test = None

        self.model = Model()

        return
    
    def retrieve_data(self, path):
        self.X, self.Y, self.Z = self.data_processor.generate_dataset(path)



    def shuffle_data(self):
        indices = np.arange(len(self.X))
        np.random.seed(2)
        self.X = [self.X[i] for i in indices]
        self.Y = [self.Y[i] for i in indices]
        self.Z = [self.Z[i] for i in indices]

    def split_data(self, train_val_test_split = [0.7, 0.9]):
        #Split the data 70 to training, 20 to validation, 10 for testing.
        split_index1 = int(train_val_test_split[0] * len(self.X))
        split_index2 = int(train_val_test_split[1] * len(self.X))

        #Train gets assigned to the first split, 70% by default
        self.X_train = self.X[:split_index1]
        self.Y_train = self.Y[:split_index1]
        self.Z_train = self.Z[:split_index1]

        #Train gets assigned to the second split, 20% by default
        self.X_val = self.X[split_index1:split_index2]
        self.Y_val = self.Y[split_index1:split_index2]
        self.Z_val = self.Z[split_index1:split_index2]

        #Train gets assigned to the last split, 10% by default
        self.X_test = self.X[split_index2: ]
        self.Y_test = self.Y[split_index2: ]
        self.Z_test = self.Z[split_index2: ]

    def evaluate(self):
        self.X_test = np.stack(self.X_test)
        self.Y_test = np.array(self.Y_test)
        loss, accuracy = self.model.neural_network.evaluate(self.X_test, self.Y_test)
        print(f"Test Accuracy: {accuracy:.4f}")

    #This function performs the entire training pipeline
    def training_pipeline(self):
        #Retrieve data.
        self.retrieve_data("../data/Data/manifest/all_data.csv")
        print (self.X[0])
        self.shuffle_data()
        self.split_data()


        layer_input = [
            {"layer_type": "Conv2D", 
             "filters" : 32,
             "kernel" : (3,3), 
              "activation": "relu" },
            {"layer_type": "Conv2D", 
             "filters" : 64,
             "kernel" : (3,3), 
              "activation": "relu" },
            {"layer_type": "Conv2D", 
             "filters" : 128,
             "kernel" : (3,3), 
              "activation": "relu" },
            {"layer_type": "Flatten"},
            {"layer_type": "Dense", 
             "units" : 128,
              "activation": "relu" },
            {"layer_type": "Dense", 
             "units" : 128,
              "activation": "softmax" }
              
        ]
        #print (type(self.X))
        self.model.build_network(layer_input)
        self.model.compile()

        self.X_train = np.stack(self.X_train)  # shape: (N, 400, 400, 3)
        self.X_val = np.stack(self.X_val)

        self.Y_train = np.array(self.Y_train)  # shape: (N,)
        self.Y_val = np.array(self.Y_val)


        self.model.train(self.X_train, self.Y_train, self.X_val, self.Y_val)

        self.evaluate()


        

        