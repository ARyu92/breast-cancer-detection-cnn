import unittest
import sys
import os
import pydicom
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing.data_processor import dataProcessor


class test_image_processor(unittest.TestCase):

    def setUp(self):
        self.processor = dataProcessor()
        #self.sample_image_path = "../data/Data/manifest/CBIS-DDSM/Calc-Test_P_00038_LEFT_CC_1/08-29-2017-DDSM-NA-94942/1.000000-ROI mask images-18515/1-1.dcm"
        self.data_directory_path = "../data/Data/manifest/"


    #Tests that this class object exists.
    def test_class_exist(self):
        self.assertIsNotNone(self.processor)

    def test_import_csv(self):
        df = self.processor.import_csv(self.data_directory_path + "mass_case_description_train_set.csv")
        
    #This tests that the the extraction of relevant columns works.
    def test_extract_columns(self):
        df = self.processor.import_csv(self.data_directory_path + "mass_case_description_train_set.csv")
        df = self.processor.extract_columns(df)
        self.assertListEqual(df.columns.tolist(), ["left or right breast", "image view", "pathology", "image file path" ])

    #This tests that a path to the image is generated from the patient row.
    def test_patient_path(self):
        df = self.processor.import_csv(self.data_directory_path + "mass_case_description_train_set.csv")
        df = self.processor.extract_columns(df)
        path = self.processor.generate_image_path(df.iloc[0])

        self.assertEqual(repr(path), repr('../data/Data/manifest/CBIS-DDSM/Mass-Training_P_00001_LEFT_CC/07-20-2016-DDSM-NA-74994\\1.000000-full mammogram images-24515\\1-1.dcm'))


    #This function tests that the an entire data sample can be assembled.
    def test_retrieve_images(self):
        df = self.processor.import_csv(self.data_directory_path + "mass_case_description_train_set.csv")
        df = self.processor.extract_columns(df)

        pixels = self.processor.retrieve_images(df.iloc[0])

    #This test tests that the min-max normalize function works.
    def test_min_max_normalize(self):
        df = self.processor.import_csv(self.data_directory_path + "mass_case_description_train_set.csv")
        df = self.processor.extract_columns(df)

        pixels = self.processor.retrieve_images(df.iloc[0])
        pixels = self.processor.min_max_normalize(pixels)
        self.assertTrue(np.all((pixels >= 0) & (pixels <= 1)))

    #This tests that the image data tensor is successfully generated.
    def test_prepare_tensor_X(self):
        df = self.processor.import_csv(self.data_directory_path + "mass_case_description_train_set.csv")
        df = self.processor.extract_columns(df)
        X, Y, Z  = self.processor.prepare_tensor(df.iloc[0])

        self.assertIsInstance(X, np.ndarray)

    #This tests that the label is generated.
    def test_prepare_tensor_Y(self):
        df = self.processor.import_csv(self.data_directory_path + "mass_case_description_train_set.csv")
        df = self.processor.extract_columns(df)
        X, Y, Z  = self.processor.prepare_tensor(df.iloc[0])

        self.assertTrue(np.all((Y == 0) | (Y == 1)))

    #This tests that the image properties is correctly generated. 
    def test_prepare_tensor_Z(self):
        df = self.processor.import_csv(self.data_directory_path + "mass_case_description_train_set.csv")
        df = self.processor.extract_columns(df)
        X, Y, Z  = self.processor.prepare_tensor(df.iloc[0])

        self.assertTrue(np.all((Z >= 0) | (Z <= 1)))

    #Tests that the dataset can be generated.
    def test_generate_dataset(self):
        X, Y, Z = self.processor.generate_dataset(self.data_directory_path + "mass_case_description_test_set.csv")

        self.assertTrue((len(X) == 50) & (len(Y) == 50) & (len(Z) == 50))

if __name__ == "__main__":
    unittest.main()

