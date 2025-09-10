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

        #Dicom file may not have these values. If these values are mpty it should hold "---"
        #Get attribute looks for the given attribute, and if not found, enters a default "" empty string.
        patient_name = getattr(dicom, "PatientName", "")
        if not patient_name:
            patient_name ="---"
            self.last_name = "---"
            self.first_name = "---"
        else: 
            patient_name = patient_name
            self.last_name = patient_name.family_name
            self.first_name = patient_name.given_name

        self.ID = getattr(dicom, "PatientID", "")
        if not self.ID :
            self.ID = "---"
                
        self.birthday = getattr(dicom, "PatientBirthDate", "")
        if not self.birthday:
            self.birthday = "---"
        
