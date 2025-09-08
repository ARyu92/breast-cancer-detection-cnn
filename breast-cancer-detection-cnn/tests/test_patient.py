import unittest
import sys
import os
import pydicom
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from patient.patient import Patient

class test_patient(unittest.TestCase):

    def setUp(self):
        self.patient = Patient()

    #This class tests that the patient class exists. 
    def test_class_exist(self):
        self.assertIsInstance(self.patient, Patient)

    #This class tests that the patient demographics can be loaded from the DICOm file
    def test_load_patient(self):
        self.patient.load_patient("C:/Users/alexr/OneDrive/London/Final Project/Source/breast-cancer-detection-cnn/data/Data/manifest/CBIS-DDSM/Calc-Test_P_00038_LEFT_CC/08-29-2017-DDSM-NA-96009/1.000000-full mammogram images-63992/1-1.dcm")
        self.assertIsNotNone(self.patient.birthday)
        self.assertIsNotNone(self.patient.last_name)
        self.assertIsNotNone(self.patient.first_name)
        self.assertIsNotNone(self.patient.ID)
if __name__ == "__main__":
    unittest.main()