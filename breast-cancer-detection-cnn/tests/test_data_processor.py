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
        file_path = "../data/meta_data_testing.csv"

        meta_data = self.processor.import_metadata(file_path)
        self.assertEqual(len(meta_data), 7)

    def test_get_uniqueIDs(self):
        data = {
            "patient_id": ["P_0001", "P_0002", "P_0001", "P_0003"],
            "label": [0, 1, 0, 1]
        }
        df = pd.DataFrame(data)
        
        uniqueIDs = self.processor.get_unique_IDs(df)
        self.assertEqual(len(uniqueIDs), 3)

    def test_get_npy_files(self):
        file_path = f'D:/Source/breast-cancer-detection-cnn/data/Processed Data/P_00001'
        tensors = self.processor.get_npy_files(file_path)
        self.assertEqual(tensors[0].shape, (512,512,2))

    def test_z_score_normalization(self):
        #Create a fake random dataset for the training, validation and testing set.
        x_train = np.random.rand(10, 32, 32, 1)
        x_val = np.random.rand(3, 32, 32, 1)
        x_test = np.random.rand(3, 32, 32, 1)

        x_train_norm, x_val_norm, x_test_norm = self.processor.z_score_normalization(x_train, x_val, x_test)

        #Mean of X_train should be just about 0, and the standard deviation just about 1.
        self.assertAlmostEqual(x_train_norm.mean(), 0.0, places = 2)
        self.assertAlmostEqual(x_train_norm.std(), 1, places = 2)

        #Normalized vald and test should not be identical to unnormalized
        self.assertFalse(np.allclose(x_val_norm, x_val))
        self.assertFalse(np.allclose(x_test_norm, x_test))

    #Tests that the tensors are returned with the proper split.
    def test_prepare_tensors(self):
        file_path = "../data/meta_data_testing.csv"
    
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

        self.assertEqual(len_X1, len_Y1)
        self.assertEqual(len_X2, len_Y2)
        self.assertEqual(len_X3, len_Y3)


if __name__ == "__main__":
    unittest.main()

