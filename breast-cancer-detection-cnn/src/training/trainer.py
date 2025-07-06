import sys
import os
import numpy as np 
import hashlib
import numpy as np 
import matplotlib.pyplot as plt

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
        X, Y, Z = self.data_processor.generate_dataset(path)
        return X, Y, Z


    def hash_array(self, arr):
        return hashlib.md5(arr.tobytes()).hexdigest()



    def shuffle_data(self, X, Y, Z, random_seed = 5):
        indices = np.arange(len(X))
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        X = [X[i] for i in indices]
        Y = [Y[i] for i in indices]
        Z = [Z[i] for i in indices]

        return X, Y, Z

    def split_data(self, X, Y, Z, train_val_test_split = [0.7, 0.9]):
        #Split the data 70 to training, 20 to validation, 10 for testing.
        split_index1 = int(train_val_test_split[0] * len(X))
        split_index2 = int(train_val_test_split[1] * len(X))

        #Train gets assigned to the first split, 70% by default
        X_train = X[:split_index1]
        Y_train = Y[:split_index1]
        Z_train = Z[:split_index1]

        #Train gets assigned to the second split, 20% by default
        X_val = X[split_index1:split_index2]
        Y_val = Y[split_index1:split_index2]
        Z_val = Z[split_index1:split_index2]

        #Train gets assigned to the last split, 10% by default
        X_test = X[split_index2: ]
        Y_test = Y[split_index2: ]
        Z_test = Z[split_index2: ]

        return X_train, Y_train, Z_train, X_val, Y_val, Z_val, X_test, Y_test, Z_test




    #This function performs the entire training pipeline
    def training_pipeline(self):
        #Retrieve data.
        self.X, self.Y, self.Z = self.retrieve_data("../data/Data/manifest/all_data.csv")

        self.X, self.Y, self.Z = self.shuffle_data(self.X, self.Y, self.Z, random_seed = 250)

        self.X_train, self.Y_train, self.Z_train, self.X_val, self.Y_val, self.Z_val, self.X_test, self.Y_test, self.Z_test = self.split_data(self.X, self.Y, self.Z)


        #self.shuffle_data()
        #self.split_data()
        self.X_train = np.stack(self.X_train)  # (N, 400, 400, 1)
        self.Y_train = np.array(self.Y_train)  # (N,)

        self.X_val = np.stack(self.X_val)
        self.Y_val = np.array(self.Y_val)

        self.X_test = np.stack(self.X_val)
        self.Y_test = np.array(self.Y_val)
        
        #for i in range(15, 17):
        #    plt.imshow(self.X[i],cmap = "gray" )
        #    plt.show()
        print("Training Set:")
        print(f"  X_train: {np.array(self.X_train).shape}")
        print(f"  Y_train: {np.array(self.Y_train).shape}")
        print(f"  Z_train: {np.array(self.Z_train).shape}")

        print("Validation Set:")
        print(f"  X_val: {np.array(self.X_val).shape}")
        print(f"  Y_val: {np.array(self.Y_val).shape}")
        print(f"  Z_val: {np.array(self.Z_val).shape}")

        print("Test Set:")
        print(f"  X_test: {np.array(self.X_test).shape}")
        print(f"  Y_test: {np.array(self.Y_test).shape}")
        print(f"  Z_test: {np.array(self.Z_test).shape}")

        print("Train:", np.bincount(self.Y_train))
        print("Val:  ", np.bincount(self.Y_val))




        layer_input = [
            {"layer_type": "Conv2D", 
             "filters" : 32,
             "kernel" : (7,7), 
              "activation": "relu" },
            {"layer_type": "Conv2D", 
             "filters" : 64,
             "kernel" : (5,5), 
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

        #print(type(self.X_train), type(self.X_train[0]))
        #print(type(self.Y_train), type(self.Y_train[0]))
        self.model.build_network(layer_input)
        self.model.compile()

        self.model.train(self.X_train, self.Y_train, self.X_val, self.Y_val)      

        self.model.evaluate(self.X_test, self.Y_test)