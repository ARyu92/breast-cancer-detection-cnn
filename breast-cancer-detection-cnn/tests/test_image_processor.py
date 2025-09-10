import unittest
import sys
import os
import pydicom
import numpy as np 
import pydicom
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing.image_processor import ImageProcessor


class test_image_processor(unittest.TestCase):

    def setUp(self):
        self.processor = ImageProcessor()
    
    #This tests checks that the import_image_tensor exists
    def test_import_image_tensor(self):
        tensor = self.processor.import_image_tensor(file_path = "blah")
        self.assertIsNone(tensor)

    #This tests checks that the import image tensor function can find a file path given a valid filepath. 
    def test_import_image_tensor_1(self):
        file_path = f'D:/Source/breast-cancer-detection-cnn/data/Processed Data/P_00001/P_00001_LEFT.npy'
        image_tensor = self.processor.import_image_tensor(file_path)

        self.assertIsNotNone(image_tensor, msg = "Image tensor holds None")

    #This test checks that the imported image tensor is of correct shape.
    def test_import_image_tensor_2(self):
        file_path = f'D:/Source/breast-cancer-detection-cnn/data/Processed Data/P_00001/P_00001_LEFT.npy'
        image_tensor = self.processor.import_image_tensor(file_path)

        expected_shape = (512,512,2)
        self.assertEqual(image_tensor.shape, expected_shape)

    def test_normalize(self):
        test_data = np.array([0, 5, 10])
        normalized = self.processor.normalize(test_data)

        np.testing.assert_array_equal(normalized, [0, 127, 255])

    #This test tests that an image can be cropped for blackspace.
    def test_crop_image(self):
        file_path = f"D:/Source/breast-cancer-detection-cnn/data/Data/manifest/CBIS-DDSM/Calc-Test_P_00038_LEFT_CC/08-29-2017-DDSM-NA-96009/1.000000-full mammogram images-63992/1-1.dcm"
        dicom = pydicom.dcmread(file_path)
        pixels = dicom.pixel_array
        pixels_cropped = self.processor.crop_image(pixels)

        #Cropped image should have less pixel columns as the original.
        self.assertLessEqual(len(pixels_cropped[0]), len(pixels[0]))

if __name__ == "__main__":
    unittest.main()

