import os 
import pandas as pd

class dataProcessor:
    def __init__(self):
        return
    
    def import_csv(self, path):
        df = pd.read_csv(path)
        return df
    
    def extract_columns(self, df):
        return df[["breast_density", "left or right breast", "image view", "pathology", "image file path"]]