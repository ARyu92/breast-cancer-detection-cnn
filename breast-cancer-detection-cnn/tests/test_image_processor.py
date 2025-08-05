import unittest
import sys
import os
import pydicom
import numpy as np 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing.image_processor import ImageProcessor


class test_image_processor(unittest.TestCase):

    def setUp(self):
        self.processor = ImageProcessor()
    
    #This tests checks that the import_image_tensor exists
    def test_import_image_tensor(self):
        self.processor.import_image_tensor(file_path = "blah")

    #This tests checks that the import image tensor function can find a file path given a valid filepath. 
    def test_import_image_tensor_1(self):
        path = "../data/Processed Data/P_00001_LEFT.npy"
        image_tensor = self.processor.import_image_tensor(path)

        self.assertIsNotNone(image_tensor, msg = "Image tensor holds None")

    #This test checks that the imported image tensor is of correct shape.
    def test_import_image_tensor_2(self):
        path = "../data/Processed Data/P_00001_LEFT.npy"
        image_tensor = self.processor.import_image_tensor(path)

        expected_shape = (2,400,400)
        self.assertEqual(image_tensor.shape, expected_shape)

if __name__ == "__main__":
    unittest.main()

