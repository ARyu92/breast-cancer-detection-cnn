import os 
import pandas as pd
import pydicom
import sys
import re
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing.image_processor import ImageProcessor

#This class is responsible for processing raw data to cleaned data.
class dataProcessor:
    def __init__(self):
        self.image_processor = ImageProcessor()
        return
    
    #Reads a CSV file and stores it into a pandas dataframe.
    def import_csv(self, path):
        df = pd.read_csv(path)
        return df
    
    #Takes the dataframe and filters by wanted properties.
    def extract_columns(self, df):
        return df[["breast_density", "left or right breast", "image view", "pathology", "image file path"]]
    
    #Generates a full image path to the image 
    def generate_image_path(self, df):
        file_path = df["image file path"]
        file_path_filter = "../data/Data/manifest/CBIS-DDSM/" + re.match(r'(.+?(CC|MLO)/)', file_path).group(0)
        extended_path = self.image_processor.find_images(file_path_filter)
        return extended_path[0]
  
    #Generates patient image data given a df row.
    def retrieve_images(self, df_row):
        image_path = self.generate_image_path(df_row)
        dicom = self.image_processor.import_image(image_path)
        pixels = self.image_processor.get_pixel_data(dicom)

        return pixels
    
    #Performs min-max normalization on a list of values
    def min_max_normalize(self, values):
        min_val = np.min(values)
        max_val = np.max(values)
        return (values - min_val) / (max_val - min_val)
    
    #This function takes in 1 row of the patient meta data and outputs an image tensor, an image description image tensor, and labels.
    #X is the image tensor, Y is the labels, Z is the image description
    def prepare_tensor(self, df_row):
        #The image
        X = self.retrieve_images(df_row)

        #The label
        label_str = df_row["pathology"]
        if label_str == "benign" :
            Y = 0 
        else: 
            Y= 1

        #The Image metadata tensor.
        Z = df_row[["breast_density", "left or right breast", "image view"]]

        #Breast density 1-4 scaled min-max
        Z["breast_density"] = (Z["breast_density"] - 1)/3

        #Left is 0, Right is 1
        if Z["left or right breast"] == "LEFT":
            Z["left or right breast"] = 0
        else:
            Z["left or right breast"] = 1

        #CC is 0, MLO is 1
        if Z["image view"] == "CC":
            Z["image view"] = 0
        else:
            Z["image view"] = 1


        return X, Y, Z

    #Extracts all data needed for model training from csv file.
    def generate_dataset(self, path):
        df = self.import_csv(path)
        df = self.extract_columns(df)

        num_of_rows = len(df)
        X = []
        Y = []
        Z = []
        for row in range(num_of_rows):
            #print(df.iloc[row])
            X_temp, Y_temp, Z_temp = self.prepare_tensor(df.iloc[row])
            X.append(X_temp)
            Y.append(Y_temp)
            Z.append(Z_temp)
        return X, Y, Z

