import unittest
import sys
import os
import pydicom
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from training.trainer import Trainer


class test_trainer(unittest.TestCase):

    def setUp(self):
        self.trainer = Trainer()
        return 
       


    #Tests that this class object exists.
    def test_class_exist(self):
        self.assertIsInstance(self.trainer, Trainer)
        return
    
    def test_split_data(self):
        X, Y, Z = self.trainer.retrieve_data("../data/Data/manifest/all_data.csv")
        X_train, Y_train, Z_train, X_val, Y_val, Z_val, X_test, Y_test, Z_test = self.trainer.split_data(X,Y,Z)
        
        training_X_len = len(X_train)
        training_Y_len = len(Y_train)
        training_Z_len = len(Z_train)

        test_X_len = len(X_test)
        test_Y_len = len(Y_test)
        test_Z_len = len(Z_test)

        val_X_len = len(X_val)
        val_Y_len = len(Y_val)
        val_Z_len = len(Z_val)

        tensor_length = len (X)
        self.assertTrue((training_X_len == tensor_length * 0.7)
                        & (training_Y_len == tensor_length* 0.7)
                        & (training_Z_len == tensor_length * 0.7)
                        & (test_X_len == tensor_length * 0.1)
                        & (test_Y_len == tensor_length * 0.1)
                        & (test_Z_len == tensor_length * 0.1)
                        & (val_X_len == tensor_length * 0.2)
                        & (val_Y_len == tensor_length * 0.2)
                        & (val_Z_len == tensor_length * 0.2)
                        
                        )

    def test_training_pipeline(self):
        self.trainer.training_pipeline()

        

if __name__ == "__main__":
    unittest.main()

