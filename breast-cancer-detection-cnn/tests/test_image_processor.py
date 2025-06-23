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
        self.sample_image_path = "../data/Data/manifest/CBIS-DDSM/Calc-Test_P_00038_LEFT_CC_1/08-29-2017-DDSM-NA-94942/1.000000-ROI mask images-18515/1-1.dcm"
        self.data_directory_path = "../data/Data/manifest/CBIS-DDSM/"


    #Tests that this class object exists.
    def test_class_exist(self):
        self.assertIsNotNone(self.processor)


    #Test a function that imports a DICOM image.
    def test_import_image(self):
        images = self.processor.find_images(self.data_directory_path)
        self.assertNotEqual(self.processor.import_image(images[0]), None)


    #Test import function returns an dicom file type.
    def test_import_image_returns_image(self):
        dicom = self.processor.import_image(self.sample_image_path)
        
        self.assertIsInstance(dicom, pydicom.dataset.FileDataset)

    #This tests that the function can take in a dicom object and return the pixel data. 
    def test_pixel_data_retrieval(self):
        dicom = self.processor.import_image(self.sample_image_path)
        image = self.processor.get_pixel_data(dicom)

        self.assertEqual(type(image), np.ndarray)


if __name__ == "__main__":
    unittest.main()

