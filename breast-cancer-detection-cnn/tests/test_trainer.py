import unittest
import sys
import os
import pydicom
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from training.trainer import Trainer


class test_image_processor(unittest.TestCase):

    def setUp(self):
        self.trainer = Trainer()
        return 
       


    #Tests that this class object exists.
    def test_class_exist(self):
        self.assertIsInstance(self.trainer, Trainer)
        return

if __name__ == "__main__":
    unittest.main()

