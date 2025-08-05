import os 
import pandas as pd
import pydicom
import sys
import re
import numpy as np
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing.image_processor import ImageProcessor

#This class is responsible for processing raw data to cleaned data.
class dataProcessor:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.meta_data = None
        return
    
    #Reads in metadata JSON file from a given file.
    def import_metadata(self, file_path):
        with open(file_path, "r") as f:
            meta_data = json.load(f)

        meta_data = np.array(meta_data)
        return meta_data


    #Prepare the tensors
    #1. Import the meta_data.
    #2. For each patient image set: 
    #       grab the image tensors and append it to X
    #       grab the label and append it to Y
    # Return X and Y
    def prepare_tensors(self, meta_data_file_path):
        self.meta_data = self.import_metadata(meta_data_file_path)
        X = []
        Y = []

        for data in self.meta_data:
            image_path = "../data/Processed Data/" + data["patient_id"] + "_" + data["laterality"] + ".npy"
            image_pair = self.image_processor.import_image_tensor(image_path)

            X.append(image_pair)

            label = data["label"]
            Y.append(label)

        X = np.array(X)
        X = X.transpose(0, 2, 3, 1)

        Y = np.array(Y)
        return X, Y
