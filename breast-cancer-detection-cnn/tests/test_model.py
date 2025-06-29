import unittest
import sys
import os
import numpy as np
from tensorflow import keras

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model.model import Model


class test_image_processor(unittest.TestCase):

    def setUp(self):
        self.model = Model()
        return 
       


    #Tests that this class object exists.
    def test_class_exist(self):
        self.assertIsInstance(self.model, Model)
        
    
    #Tests that the network has the specified number of layers.
    def test_class_exist(self):
        #List holds each layer's parameters in a dictionary object.
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
        self.model.build_network(layer_input)
        num_of_layers = len(self.model.neural_network.layers)

        self.assertEqual(num_of_layers, 9)


if __name__ == "__main__":
    unittest.main()

