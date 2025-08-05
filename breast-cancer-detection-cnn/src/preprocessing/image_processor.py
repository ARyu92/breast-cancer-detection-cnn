import os
import pydicom
import numpy as np
import cv2

class ImageProcessor:
    def __init__(self):
        return
    
    #This takes in a file path to a .npy file which contains the preprocessed tensor, and returns the tensor
    def import_image_tensor(self, file_path):
        try:
            image_tensor = np.load(file_path)
        except:
            print (f"File failed to load, check that the path: {file_path} is correct")
            return

        return image_tensor
    
    
