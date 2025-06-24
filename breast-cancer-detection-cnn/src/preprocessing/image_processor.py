import os
import pydicom
import numpy
import cv2

class ImageProcessor:
    def __init__(self):
        return
    
    #This function finds all images in a given directory.
    def find_images(self, root_dir):
        dicom_paths = []

        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(".dcm"):
                    full_path = os.path.join(dirpath, filename)
                    dicom_paths.append(full_path)
        return dicom_paths


    #This function takes in a file path to a dicom image and outputs an object holding the dicom file data.
    def import_image(self, file_path):
        dicom_file = pydicom.dcmread(file_path)
        return dicom_file

    #Returns the pixel data from the dicom object. 
    def get_pixel_data(self, dicom):
        pixels = dicom.pixel_array
        return pixels
    
    #Resizes the pixel data to a given input size.
    def resize_image(self, pixels, dimensions = (400,400)):
        pixels = cv2.resize(pixels, dimensions)
        return pixels