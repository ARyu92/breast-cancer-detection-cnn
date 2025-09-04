import unittest
import sys
import os
import pydicom
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from training.trainer import Trainer


class test_trainer(unittest.TestCase):

    def setUp(self):
        self.trainer = Trainer()
        return 
       
    #Tests that this class object exists.
    def test_class_exist(self):
        self.assertIsInstance(self.trainer, Trainer)
        return

                    

    def test_training_pipeline(self):
        self.trainer.training_pipeline(filter1= 64, filter2= 128, filter3= 256, epochs = 100)
        return

    def grid_search_training(self):
        #Row 1: 64, 128, 256 filters in first three, each time going up a magnitude on the class weighting factors.
        self.trainer.training_pipeline(filter1= 64, filter2= 128, filter3= 256, 
                                       class_weight_benign= 5, class_weight_malignant= 95)
        
        self.trainer.training_pipeline(filter1= 64, filter2= 128, filter3= 256, 
                                       class_weight_benign= 10, class_weight_malignant= 90)
        
        self.trainer.training_pipeline(filter1= 64, filter2= 128, filter3= 256, 
                                       class_weight_benign= 20, class_weight_malignant= 80)


        #Row 2: Half the filter progression as row 1, each time going up a magnitude on the class weighting factors.
        self.trainer.training_pipeline(filter1= 32, filter2= 64, filter3= 128, 
                                       class_weight_benign= 5, class_weight_malignant= 95)
        
        self.trainer.training_pipeline(filter1= 32, filter2= 64, filter3= 128, 
                                       class_weight_benign= 10, class_weight_malignant= 90)
        
        self.trainer.training_pipeline(filter1= 32, filter2= 64, filter3= 128, 
                                       class_weight_benign= 20, class_weight_malignant= 80)


if __name__ == "__main__":
    unittest.main()

