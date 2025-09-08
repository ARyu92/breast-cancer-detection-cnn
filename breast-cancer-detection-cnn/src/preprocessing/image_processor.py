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
    
    def extract_pixels(self, dicom):
        pixels = dicom.pixel_array
        return pixels
    
    def normalize(self, pixels):
        pixels = pixels.astype(float)
        pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min()) * 255
        pixels = pixels.astype(np.uint8)
        return pixels
    
    #This function takes in an image and crops the black space out.
    def crop_image(self, image, patch_size = 40,
                                black_low: int = 0, black_high =  300,
                                patch_black_ratio = 0.75):

        height, width = image.shape
        # Patch_size minus 1, to start from index 0 
        num_rows_patches = (height + patch_size - 1) // patch_size
        num_columns_patches = (width + patch_size - 1) // patch_size

        patch_black = np.zeros((num_rows_patches, num_columns_patches), dtype=bool)
        
        for i in range(num_rows_patches):
            y1 = i * patch_size
            #In case the array runs out of bounds, use the smaller value between calculated and total image height.
            y2 = min((i + 1) * patch_size, height)
            
            for j in range(num_columns_patches):
                x1 = j * patch_size 
                x2 = min((j + 1) * patch_size, width)
                patch = image[y1:y2, x1:x2]

                frac_in_range = np.mean((patch >= black_low) & (patch <= black_high))
                patch_black[i, j] = (frac_in_range >= patch_black_ratio)

        col_black = patch_black.all(axis=0)

        keep_pixel_cols = np.ones(width, dtype=bool)
        for j, is_black in enumerate(col_black):
            if is_black:
                x1 = j * patch_size
                #In case the array runs out of bounds, use the smaller value between calculated and total image height.
                x2 = min((j + 1) * patch_size, width)
                keep_pixel_cols[x1:x2] = False

        #Need a safety clause, in case no columns are removed.
        if not keep_pixel_cols.any():
            return image

        return image[:, keep_pixel_cols]
    
    
