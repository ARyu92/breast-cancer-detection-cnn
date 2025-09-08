import pydicom
from datetime import datetime

class Patient():
    def __init__(self):
        self.first_name = None
        self.last_name = None
        self.birthday = None
        self.ID = None

    #This function takes in a dicom path to a patient and loads demographic data. 
    def load_patient(self, dicom_path):
        dicom = pydicom.dcmread(dicom_path)
        patient_name = dicom.PatientName
        self.last_name = patient_name.family_name
        self.first_name = patient_name.given_name
        self.ID = dicom.PatientID
        self.birthday = dicom.PatientBirthDate
