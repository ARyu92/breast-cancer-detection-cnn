import unittest
import sys
import os
import pydicom
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model.model import Model


class test_image_processor(unittest.TestCase):

    def setUp(self):
        self.model = Model()
        return 
       


    #Tests that this class object exists.
    def test_class_exist(self):
        self.assertIsInstance(self.model, Model)
        return

if __name__ == "__main__":
    unittest.main()

