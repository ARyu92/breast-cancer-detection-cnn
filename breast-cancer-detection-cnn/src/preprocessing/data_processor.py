import os 
import pandas as pd
import pydicom
import sys
import re
import numpy as np
import json
import io
import cv2 

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
    
    #This processes a StreamLit uploadedFile object to a dicom file
    def uploadedfile_to_dicom(self, uploaded_file):
        if uploaded_file is None:
            raise ValueError("No file uploaded.")
        
        # Read the uploaded file's bytes into memory
        file_bytes = uploaded_file.read()
        
        # Convert bytes into a file-like object
        file_stream = io.BytesIO(file_bytes)
        
        # Read into a pydicom Dataset
        dicom_dataset = pydicom.dcmread(file_stream)
        
        return dicom_dataset

    #This function takes in a DICOM file and extracts pixel data from it 
    def extract_pixels_from_dicom(self, dicom):
        pixel_array = dicom.pixel_array.astype(np.float32)

        # Normalize to 0-255 range
        pixel_array -= pixel_array.min()
        if pixel_array.max() > 0:
            pixel_array /= pixel_array.max()
        pixel_array *= 255

        pixel_array = cv2.resize(pixel_array, [400,400])

        # Convert to uint8
        return pixel_array.astype(np.uint8)
    
