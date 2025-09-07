import unittest
import sys
import os
import numpy as np
from tensorflow import keras
import shutil
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model.model import Model


class test_image_processor(unittest.TestCase):

    def setUp(self):
        self.model = Model()
        self.model.build_split_network()
        return 
       
    #Tests that this class object exists.
    def test_class_exist(self):
        self.assertIsInstance(self.model, Model)
        
    
    #Tests that the network has the specified number of layers.
    def test_class_exist(self):
        num_of_layers = len(self.model.neural_network.layers)

        self.assertEqual(num_of_layers, 17)

    #Checks that the model saves to a given path.
    def test_save_model(self):
        self.model.save_model("../trained_models/", "model_test")
        path_exists = os.path.exists("../trained_models/model_test/model_test.keras")

        self.assertTrue(path_exists)

        #Delete the created model to clean up for test re-runs.
        shutil.rmtree("../trained_models/model_test/")
        
    #Tests that a model created with the same name has an integer incremented to it
    def test_save_model(self):
        self.model.save_model("../trained_models/", "model_test")
        self.model.save_model("../trained_models/", "model_test")
        self.model.save_model("../trained_models/", "model_test")

        path_exists = os.path.exists("../trained_models/model_test/model_test.keras")
        path_exists1 = os.path.exists("../trained_models/model_test_1/model_test_1.keras")
        path_exists2 = os.path.exists("../trained_models/model_test_2/model_test_2.keras")

        self.assertTrue(path_exists and path_exists1 and path_exists2)

        #Delete the created models to clean up for test re-runs.
        shutil.rmtree("../trained_models/model_test/")
        shutil.rmtree("../trained_models/model_test_1/")
        shutil.rmtree("../trained_models/model_test_2/")

        

    def test_load_model(self):
        self.model.save_model("../trained_models/", "model_load_test")
        loaded_model = self.model.load_model("../trained_models/model_load_test/model_load_test.keras")
        num_of_layers = len(loaded_model.layers)

        loaded_model.summary()

        self.assertEqual(num_of_layers, 17)

    def test_temp_model_save(self):
        path_name = self.model.temp_save_model("1A")
        print(path_name)
if __name__ == "__main__":
    unittest.main()

