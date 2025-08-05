import unittest
import sys
import os
import pydicom
import numpy as np

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
        file_path = "../data/meta_data.json"

        meta_data = self.processor.import_metadata(file_path)

    #Tests that all elements of the metadata are imported.
    def test_import_metadata_1(self):
        file_path = "../data/meta_data.json"
        meta_data = self.processor.import_metadata(file_path)

        self.assertEqual(len(meta_data), 1142)

    #Checks that each dictionary of the meta data array has keys: patient_id, laterality, cc_path, mlo_path, and label
    def test_import_metadata_2(self):
        file_path = "../data/meta_data.json"
        meta_data = self.processor.import_metadata(file_path)

        key_exists = True
        for data in meta_data:
            keys = data.keys()
            if ("patient_id" not in keys 
                or "laterality" not in keys
                or "cc_path" not in keys
                or "mlo_path" not in keys
                or "label" not in keys) :
                key_exists = False
                break
        self.assertTrue(key_exists)

    #Checks that the function returns image tensors and label tensors.
    def test_prepare_tensors(self):
        X, Y = self.processor.prepare_tensors("../data/meta_data.json")
        self.assertEqual(len(X), len(Y))
        self.assertEqual(len(X), 1142)
        




if __name__ == "__main__":
    unittest.main()

