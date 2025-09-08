import os 
import pandas as pd
import pydicom
import sys
import re
import numpy as np
import io
import cv2 
import random
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing.image_processor import ImageProcessor

#This class is responsible for processing raw data to cleaned data.
class dataProcessor:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.meta_data = None
        return
    
    #Reads in metadata CSV file from a given file.
    def import_metadata(self, file_path):
        meta_data = pd.read_csv(file_path)
        return meta_data
    
    def get_unique_IDs(self, df):
        uniqueIDs = df["patient_id"].unique()
        return uniqueIDs
    
    def get_npy_files(self, directory_path):
        npy_files = []
        tensors = []
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(".npy"):
                npy_files.append(os.path.join(directory_path, filename))

        for file in npy_files:
            tensor = np.load(file)
            tensors.append(tensor)
        return tensors


    #Prepare the tensors
    #1. Import the meta data to pandas DF
    #2. Get a list of unique patient IDs from the meta data csv
    #3. Shuffle the list of unique patient IDs
    #4. Split the list of unique patient IDs by into training, validation and testing using the split ratios
    #5  for each set, search for each patient's ID on the meta data. 
    ### It could be that there are multiple rows returned, because the same patient can have more than 1 image set.
    #Code should look somewhat like:
    #. For each unique_patient_ID in unique_patient_IDs:
        #Get row_slices of the meta_data that has unique_patient_ID
        #For each row in row_slices
            #Append image data
            #Append label data
    def prepare_tensors(self, meta_data_file_path, split=[0.70, 0.20, 0.10], seed=509):
        #1. Import the meta data
        self.meta_data = self.import_metadata(meta_data_file_path)

        #2. Get a list of unique patient IDs from the meta data csv
        unique_patient_IDs = self.get_unique_IDs(self.meta_data)
        total_patients = len(unique_patient_IDs)


        #3. Shuffle the list of unique patient IDs
        random.seed(seed)
        random.shuffle(unique_patient_IDs)

        #4. Split the list of unique patient IDs by into training, validation and testing using the split ratios
        n_train = int(total_patients * split[0])
        n_val   = int(total_patients * split[1])
       
        train_patients = unique_patient_IDs[:n_train]
        val_patients   = unique_patient_IDs[n_train:n_train+n_val]
        test_patients  = unique_patient_IDs[n_train+n_val:]

        X_train, Y_train = [], []
        X_val, Y_val = [], []
        X_test, Y_test = [], []

        def load(patients, X, Y):
            #search for each patient's ID on the meta data. 
            for pid in patients:
                rows = self.meta_data[self.meta_data["patient_id"] == pid]

                #For each row in row_slices
                for _, row in rows.iterrows():
                    #Have to allow pickle to enable loading object arrays.
                    tensor = np.load(row["tensor_path"], allow_pickle=True)

                    #Append image data
                    #Append label data
                    X.append(tensor)
                    Y.append(row["label"])

        # Fill splits
        load(train_patients, X_train, Y_train)
        load(val_patients,   X_val,   Y_val)
        load(test_patients,  X_test,  Y_test)

        # Convert to numpy arrays, so that it is in format accepted by the keras model. 
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_val   = np.array(X_val)
        Y_val   = np.array(Y_val)
        X_test  = np.array(X_test)
        Y_test  = np.array(Y_test)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    #This function will take in an array of image tensors X_train, X_Val, X_test and applies Z score normalization.
    def z_score_normalization(self, X_train, X_val, X_test, validate = True):
        mean = X_train.mean(axis=(0, 1, 2), keepdims=True).astype(np.float32)
        std  = (X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-6).astype(np.float32)

<<<<<<< HEAD
        # Apply z-score normalization to all sets
        X_train = (X_train - mean) / std
        if (validate):
            X_val   = (X_val   - mean) / std
        X_test  = (X_test  - mean) / std

        return X_train, X_val, X_test
=======
    
    #This function takes in a path to a dicom file and extracts patient data from it.
    def process_dicom(self, dicom_path):
        dicom = pydicom.dcmread(dicom_path)
        pixels = self.image_processor.extract_pixels(dicom)
        pixels = self.image_processor.normalize(pixels)
        return pixels
    
    #This function takes in a path to a dicom file and processes it into a tensor.
    def process_image(self, dicom_path):
        dicom = pydicom.dcmread(dicom_path)
        pixels = self.image_processor.extract_pixels(dicom)

        pixels = apply_voi_lut(pixels, dicom)
        pixels = cv2.resize(pixels, (512,512))
        pixels = self.image_processor.crop_image(pixels, patch_size = 32,
                                        black_low=0, black_high=100,
                                        patch_black_ratio=0.6)
        pixels = cv2.resize(pixels, (512,512))

        return pixels


>>>>>>> origin/GUI/Finalization
