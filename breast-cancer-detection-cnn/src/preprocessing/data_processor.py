import os 
import pandas as pd
import pydicom
import sys
import re
import numpy as np
import io
import cv2 
import random

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
    #1. Import the meta data
    #2. Get a list of unique patient IDs from the meta data csv
    #3. For each unique_patient_ID in unique_patient_IDs:
        #Get row_slices of the meta_data that has unique_patient_ID
        #For each row in row_slices
            #Append image data
            #Append label data


    def prepare_tensors(self, meta_data_file_path, split=[0.70, 0.20, 0.10], seed=42):
        self.meta_data = self.import_metadata(meta_data_file_path)

        # Unique patients
        unique_patient_IDs = self.get_unique_IDs(self.meta_data)
        total_patients = len(unique_patient_IDs)

        # Shuffle once
        random.seed(seed)
        random.shuffle(unique_patient_IDs)

        # Patient-level split
        n_train = int(total_patients * split[0])
        n_val   = int(total_patients * split[1])
        # rest go to test
        train_patients = unique_patient_IDs[:n_train]
        val_patients   = unique_patient_IDs[n_train:n_train+n_val]
        test_patients  = unique_patient_IDs[n_train+n_val:]

        X_train, Y_train = [], []
        X_val, Y_val = [], []
        X_test, Y_test = [], []

        def load(patients, X, Y):
            for pid in patients:
                rows = self.meta_data[self.meta_data["patient_id"] == pid]
                for _, row in rows.iterrows():
                    tensor = np.load(row["tensor_path"], allow_pickle=True)
                    X.append(tensor)
                    Y.append(row["label"])

        # Fill splits
        load(train_patients, X_train, Y_train)
        load(val_patients,   X_val,   Y_val)
        load(test_patients,  X_test,  Y_test)

        # Convert to numpy arrays
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_val   = np.array(X_val)
        Y_val   = np.array(Y_val)
        X_test  = np.array(X_test)
        Y_test  = np.array(Y_test)


        return X_train, Y_train, X_val, Y_val, X_test, Y_test


