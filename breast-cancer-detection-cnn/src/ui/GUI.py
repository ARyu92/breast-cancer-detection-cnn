import streamlit as st
import numpy as np

class GUI:
    def __init__(self):
        st.set_page_config(layout="wide")
        # Reduce top and bottom padding
        st.markdown(
            """
            <style>
            .block-container {
                padding-top: 0rem;
                padding-bottom: 1rem;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

#--------------------------------------------------------------Header--------------------------------------------------------#
        header = st.container()
        with header:
            st.markdown("## Breast Cancer Detection")
            st.markdown("An AI Powered Application")


#--------------------------------------------------------------Navigator-----------------------------------------------------#
        tool_pad1, tool_col1, tool_col2, tool_pad2,  = st.columns([1.5, 1, 1, 6.5])

        with tool_col1:
            if st.button("Load Image", use_container_width= True, key = "load_immage_button"):
                self.load_image_button()

        with tool_col2:
            st.button("Load Model", use_container_width= True, key = "load_model_button")

#--------------------------------------------------------------Main Body-----------------------------------------------------#
        info_bar, cc_image_holder, body_pad1 ,mlo_image_holder, body_pad2 = st.columns([1.5,3.5, .2, 3.5, 0.8])

        with info_bar: 
            st.subheader("Patient:")
            name_box = st.empty()
            name = "example"
            
            name_box.write(f"Patient Name: {name}")

        placeholder_img = np.zeros((400, 400, 3), dtype=np.uint8)
        with cc_image_holder:
            cc_image_slot = st.empty()
            cc_image_slot.image(image = placeholder_img, caption="Craniocaudal", use_container_width =True)

        with mlo_image_holder:
            mlo_image_slot = st.empty()
            mlo_image_slot.image(image = placeholder_img, caption= "Mediolateral Oblique", use_container_width =True)

#--------------------------------------------------------------Input_output-------------------------------------------------#
        model_options, console, malignant_marker, bengin_marker, foot_pad1 = st.columns([1.5, 6.5, 0.75 ,0.75, 0.5])

        with model_options:
            st.write("Model Options")
            st.button("⚙️")

        with console:
            output_text = "sample console output"
            st.write("Console")

            st.code(output_text)

        with malignant_marker:
            st.markdown("")
            st.markdown("")
            st.button("Malignant", use_container_width= True, key = "malignant_marker_button")

        with bengin_marker:
            st.markdown("")
            st.markdown("")
            st.button("Benign", use_container_width= True, key = "benign_marker_button")

    def load_image_button(self):
        placeholder_img = st.file_uploader("Upload an image in DICOM format", type = ["dcm"])
