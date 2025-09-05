import sys
import os
from qtpy import QtWidgets, QtCore, QtGui
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from preprocessing.data_processor import dataProcessor
from preprocessing.image_processor import ImageProcessor
from model.model import Model




#The GUI is an inheirited class of the QtWidgets class.
class GUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.output_text = ""
        self.neural_network = Model()
        self.data_processor = dataProcessor()
        self.image_processor = ImageProcessor()
        self.cc_tensor = None
        self.mlo_tensor = None


        self.apply_dark_theme()

        self.setWindowTitle("Breast Cancer Detection App")
        self.resize(1920,1080)
        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)


        #Header---------------------------------------------------------
        self.header = self.define_header()
        self.main_layout.addLayout(self.header)

        #Navigator----------------------------------------------------
        self.navigator = self.define_navigator()
        self.main_layout.addLayout(self.navigator)

        #Body----------------------------------------------------------
        self.body = QtWidgets.QHBoxLayout()
        #Info bar
        self.info_bar = self.define_info_bar()
        self.body.addLayout(self.info_bar)

        #Image Display
        self.image_display = self.define_image_display()
        self.body.addLayout(self.image_display)

        #Add body layout to main layout.
        self.main_layout.addLayout(self.body)

        #Footer-------------------------------------------------------
        self.footer = self.define_footer()
        self.main_layout.addLayout(self.footer)

        self.main_layout.addStretch()
    
    #This function applies a dark color theme
    def apply_dark_theme(self):
        #To do
        return 
    
    def define_header(self):
        header = QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel("Breast Cancer Detection App")

        header.addWidget(label)

        return header
    
    def define_navigator(self):
        navigator = QtWidgets.QHBoxLayout()
        navigator.addStretch(1.5)

        #Initialize load CC image button and add to nanvigator.
        self.load_CC_image_button = QtWidgets.QPushButton("Load CC Image")
        navigator.addWidget(self.load_CC_image_button)
        self.load_CC_image_button.clicked.connect(self.on_load_CC_image_button_clicked)

        #Initialize load MLO image button and add to nanvigator.
        self.load_MLO_image_button = QtWidgets.QPushButton("Load MLO Image")
        navigator.addWidget(self.load_MLO_image_button)
        self.load_MLO_image_button.clicked.connect(self.on_load_MLO_image_button_clicked)


        #Initialize load model button as above.
        self.load_model_button = QtWidgets.QPushButton("Load Model")
        navigator.addWidget(self.load_model_button)
        self.load_model_button.clicked.connect(self.on_load_model_button_clicked)



        self.option_button = QtWidgets.QPushButton("Options")
        navigator.addWidget(self.option_button)
        self.option_button.clicked.connect(self.on_option_button_clicked)

        navigator.addStretch(8.5)


        return navigator
    
    def define_info_bar(self):
        info_bar = QtWidgets.QVBoxLayout()

        #Individual patient data
        #Patient ID
        self.patient_id = QtWidgets.QLabel(f"Patient ID: ---")
        info_bar.addWidget(self.patient_id)

        #Patient First Name
        self.patient_fname = QtWidgets.QLabel(f"First Name: ---")
        info_bar.addWidget(self.patient_fname)

        #Patient Last Name
        self.patient_lname = QtWidgets.QLabel(f"Last Name: ---")
        info_bar.addWidget(self.patient_lname)

        #Birthday
        self.patient_birthday = QtWidgets.QLabel(f"Birthday: ---")
        info_bar.addWidget(self.patient_birthday)

        #Add spacer
        info_bar.addStretch(15)

        return info_bar
    
    def update_patient_info(self):
        patient_id_value = "Test233"
        self.patient_id.setText(f"Patient ID: {patient_id_value}")

        patient_fname_value = "Fran"
        self.patient_fname.setText = (f"First Name: {patient_fname_value}")

        patient_lname_value = "Davis"
        self.patient_lname.setText = (f"Last Name: {patient_lname_value}")

        patient_birthday_value = "July 22 1992"
        self.patient_birthday.setText = (f"Birthday: {patient_birthday_value}")

    def define_image_display(self):
        image_display = QtWidgets.QHBoxLayout()
        #Add spacers on the left
        image_display.addStretch(1)
        # Left image box
        self.image_box_left = QtWidgets.QLabel()
        self.image_box_left.setFixedSize(800, 800)
        self.image_box_left.setStyleSheet("background-color: black; border: 2px solid white;")
        self.image_box_left.setAlignment(QtCore.Qt.AlignCenter)

        # Right image box
        self.image_box_right = QtWidgets.QLabel()
        self.image_box_right.setFixedSize(800, 800)
        self.image_box_right.setStyleSheet("background-color: black; border: 2px solid white;")
        self.image_box_right.setAlignment(QtCore.Qt.AlignCenter)

        image_display.addWidget(self.image_box_left)
        image_display.addWidget(self.image_box_right)
        image_display.addStretch(1)

        return image_display

    def define_footer(self):
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)

        # --- Text output area ---
        self.output_box = QtWidgets.QLabel("Output: ---")
        self.output_box.setFixedHeight(100)
        self.output_box.setStyleSheet("background-color: #1e1e1e; color: white; border: 1px solid white; padding: 8px;")
        self.output_box.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        # Make it expand horizontally (take most of the footer width)
        footer.addWidget(self.output_box, stretch=6)

        # --- Buttons ---
        self.benign_button = QtWidgets.QPushButton("Benign")
        self.malignant_button = QtWidgets.QPushButton("Malignant")

        # Keep buttons compact
        self.benign_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.malignant_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        #Connect buttons to event handlers.
        self.benign_button.clicked.connect(self.on_benign_button_clicked)
        self.malignant_button.clicked.connect(self.on_malignant_button_clicked)

        footer.addWidget(self.benign_button, stretch=0)
        footer.addWidget(self.malignant_button, stretch=0)

        return footer
    
    def on_benign_button_clicked(self):
        self.make_prediction()

    def on_malignant_button_clicked(self):
        self.make_prediction()

    def on_load_model_button_clicked(self):
        self.output_box.setText("Load model button clicked")
        model_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Model", "", "DICOM Files (*.keras)"
        )
        if not model_path:
            return

        self.neural_network.load_model(model_path)
        if (self.neural_network.neural_network is not None):
            self.output_box.setText("Model loaded")

        self.neural_network.neural_network.compile()

    def on_load_CC_image_button_clicked(self):
        self.output_box.setText("Load CC image button clicked")

        dicom_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Patient DICOM file", "", "DICOM Files (*.dcm)"
        )
        if not dicom_path:
            return

        pixels = self.data_processor.process_dicom(dicom_path)


        # build QImage from NumPy data
        h, w = pixels.shape
        qimg = QtGui.QImage(pixels.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        qimg = qimg.copy()

        # convert to QPixmap and show on your label
        pixmap = QtGui.QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(
            self.image_box_left.size(),
            QtCore.Qt.KeepAspectRatio
        )
        self.image_box_left.setPixmap(pixmap)

        #Create the tensor.
        self.cc_tensor = self.data_processor.process_image(dicom_path)

    def on_load_MLO_image_button_clicked(self):
        self.output_box.setText("Load MLO image button clicked")

        dicom_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Patient DICOM file", "", "DICOM Files (*.dcm)"
        )
        if not dicom_path:
            return

        pixels = self.data_processor.process_dicom(dicom_path)


        # build QImage from NumPy data
        h, w = pixels.shape
        qimg = QtGui.QImage(pixels.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        qimg = qimg.copy()

        # convert to QPixmap and show on your label
        pixmap = QtGui.QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(
            self.image_box_right.size(),
            QtCore.Qt.KeepAspectRatio
        )
        self.image_box_right.setPixmap(pixmap)

        #Create the tensor.
        self.mlo_tensor = self.data_processor.process_image(dicom_path)


    def on_option_button_clicked(self):
        self.output_box.setText("Option button clicked")
        
    def make_prediction(self):
        self.output_text = ""
        is_ready_for_prediction = True
        if self.neural_network.neural_network == None:
            self.output_text += "No Model loaded.\n"
            is_ready_for_prediction = False
        
        if self.cc_tensor is None:
            self.output_text += "No CC image loaded. \n"
            is_ready_for_prediction = False
        
        if self.mlo_tensor is None:
            self.output_text += "No MLO image loaded. \n"
            is_ready_for_prediction = False

        #If tensors and model are not loaded, then output message to user and return
        if (not is_ready_for_prediction):
            self.output_box.setText(self.output_text)
            return
        
        stacked_tensor = np.stack([self.cc_tensor, self.mlo_tensor], axis=-1)
        stacked_tensor = np.expand_dims(stacked_tensor, axis=0)

        prediction = self.neural_network.neural_network.predict(stacked_tensor)

        if (prediction == 0):
            self.output_text += "Model predicts Benign\n"
        else:
            self.output_text += "Model predicts Malignant\n"
        
        self.output_box.setText(self.output_text)


        
        

        





