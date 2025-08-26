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

                    

    def test_training_pipeline(self):
        self.trainer.training_pipeline(filter1= 64, filter2= 128, filter3= 256)

        

if __name__ == "__main__":
    unittest.main()

