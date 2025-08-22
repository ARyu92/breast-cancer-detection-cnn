import unittest
import sys
import os
import pydicom
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing.data_processor import dataProcessor


class test_data_processor(unittest.TestCase):

    def setUp(self):
        self.processor = dataProcessor()


    #Tests that this class object exists.
    def test_class_exist(self):
        self.assertIsNotNone(self.processor)

    #Tests that the metadata file can be imported given a valid file path. 
    def test_import_metadata(self):
        file_path = "../data/meta_data.csv"

        meta_data = self.processor.import_metadata(file_path)
        self.assertEqual(len(meta_data), 1324)

    def test_get_uniqueIDs(self):
        data = {
            "patient_id": ["P_0001", "P_0002", "P_0001", "P_0003"],
            "label": [0, 1, 0, 1]
        }
        df = pd.DataFrame(data)
        
        uniqueIDs = self.processor.get_unique_IDs(df)
        self.assertEqual(len(uniqueIDs), 3)

    #Tests that the tensors are returned with the proper split.
    def test_prepare_tensors(self):
        file_path = "../data/meta_data.csv"
    
        X1, Y1, X2, Y2, X3, Y3 = self.processor.prepare_tensors(file_path)
        
        len_X1 = len(X1)
        len_Y1 = len(Y1)
        len_X2 = len(X2)
        len_Y2 = len(Y2)
        len_X3 = len(X3)
        len_Y3 = len(Y3)

        print("Train set:", len_X1, len_Y1)
        print("Val set:", len_X2, len_Y2)
        print("Test set:", len_X3, len_Y3)

        # Optional sanity checks
        self.assertEqual(len_X1, len_Y1)
        self.assertEqual(len_X2, len_Y2)
        self.assertEqual(len_X3, len_Y3)

if __name__ == "__main__":
    unittest.main()

