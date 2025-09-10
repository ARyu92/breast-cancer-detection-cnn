import unittest
import sys
import os
import numpy as np
from tensorflow import keras
import shutil
import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model.model import Model


class test_model(unittest.TestCase):

    def setUp(self):
        self.model = Model()
        self.model.build_split_network()
        return 
       
    #Tests that this class object exists.
    def test_class_exist(self):
        self.assertIsInstance(self.model, Model)
        
    
    #Tests that the network has the specified number of layers.
    def test_class_exist_1(self):
        num_of_layers = len(self.model.neural_network.layers)

        self.assertEqual(num_of_layers, 14)

    #Checks that the model saves to a given path.
    def test_save_model(self):
        self.model.save_model("model_test", training_mean = 3.14, training_std= 1)
        path_exists = os.path.exists("../trained_models/model_test/model_test.h5")

        self.assertTrue(path_exists)

        #Delete the created model to clean up for test re-runs.
        shutil.rmtree("../trained_models/model_test/")
        
    #Tests that a model created with the same name has an integer incremented to it
    def test_save_model_1(self):
        self.model.save_model("model_test", training_mean = 3.14, training_std= 1)
        self.model.save_model("model_test", training_mean = 3.2, training_std= 2)
        self.model.save_model("model_test", training_mean = 3.3, training_std= 1.5)

        path_exists = os.path.exists("../trained_models/model_test/model_test.h5")
        path_exists1 = os.path.exists("../trained_models/model_test_1/model_test_1.h5")
        path_exists2 = os.path.exists("../trained_models/model_test_2/model_test_2.h5")

        self.assertTrue(path_exists and path_exists1 and path_exists2)

        #Delete the created models to clean up for test re-runs.
        shutil.rmtree("../trained_models/model_test/")
        shutil.rmtree("../trained_models/model_test_1/")
        shutil.rmtree("../trained_models/model_test_2/")

    #This tests that a created neural network model can be loaded in.
    def test_load_model(self):
        self.model.save_model("model_load_test_load", training_mean = 3.14, training_std= 1)
        self.model.load_model("../trained_models/model_load_test_load/model_load_test_load.h5")
        num_of_layers = len(self.model.neural_network.layers)

        self.assertEqual(num_of_layers, 14)
        shutil.rmtree("../trained_models/model_load_test_load/")

    #This tests that a created neural network comes back with the correct assigned attribute.
    def test_load_model_1(self):
        self.model.save_model("model_load_test_load", training_mean = 3.14, training_std= 1)
        mean, standard_dev = self.model.load_model("../trained_models/model_load_test_load/model_load_test_load.h5")
        
        self.assertEqual(mean, 3.14)
        self.assertEqual(standard_dev, 1)
        shutil.rmtree("../trained_models/model_load_test_load/")

if __name__ == "__main__":
    unittest.main()

