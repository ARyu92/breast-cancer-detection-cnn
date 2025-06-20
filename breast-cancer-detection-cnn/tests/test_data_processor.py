import unittest
import sys
import os
import pydicom

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
        

    def test_extract_columns(self):
        df = self.processor.import_csv(self.data_directory_path + "mass_case_description_train_set.csv")
        df = self.processor.extract_columns(df)
        print (df)
        self.assertListEqual(df.columns.tolist(), ["breast_density", "left or right breast", "image view", "pathology", "image file path" ])


if __name__ == "__main__":
    unittest.main()

