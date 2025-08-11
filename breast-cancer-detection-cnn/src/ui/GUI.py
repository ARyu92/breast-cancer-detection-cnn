import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import streamlit as st
import numpy as np
from preprocessing.data_processor import dataProcessor


class GUI:
    def __init__(self):
        self.data_processor = dataProcessor()

        # --- Page layout ---
        st.set_page_config(layout="wide")
        st.markdown("""
            <style>
            .block-container { padding-top: 0rem; padding-bottom: 1rem; }
            </style>
        """, unsafe_allow_html=True)

        # --- Session state ---
        if "show_uploader" not in st.session_state:
            st.session_state.show_uploader = False
        if "uploader_counter" not in st.session_state:
            st.session_state.uploader_counter = 0
        # store last displayed CC image
        if "cc_last_img" not in st.session_state:
            st.session_state.cc_last_img = None

        # --- Build UI ---
        self.build_header()
        self.build_toolbar()
        self.build_main_body()
        self.build_footer()

        # --- Conditionally show uploader ---
        if st.session_state.show_uploader:
            self.show_uploader()

    def build_header(self):
        with st.container():
            st.markdown("## Breast Cancer Detection")
            st.markdown("An AI Powered Application")

    def build_toolbar(self):
        tool_pad1, tool_col1, tool_col2, tool_pad2 = st.columns([1.5, 1, 1, 6.5])

        with tool_col1:
            # Always visible button
            if st.button("Load Image", use_container_width=True, key="load_image_button"):
                st.session_state.show_uploader = True
                # bump key so uploader gets recreated each time
                st.session_state.uploader_counter += 1

        with tool_col2:
            st.button("Load Model", use_container_width=True, key="load_model_button")

    def build_main_body(self):
        info_bar, cc_image_holder, body_pad1, mlo_image_holder, body_pad2 = st.columns([1.5, 3.5, .2, 3.5, 0.8])

        with info_bar:
            st.subheader("Patient:")
            st.write("Patient Name: example")

        placeholder_img = np.zeros((400, 400, 3), dtype=np.uint8)

        with cc_image_holder:
            self.cc_image_slot = st.empty()
            # Persisted image if available; otherwise placeholder
            if st.session_state.cc_last_img is not None:
                self.cc_image_slot.image(
                    st.session_state.cc_last_img,
                    caption="Craniocaudal",
                    use_container_width=True
                )
            else:
                self.cc_image_slot.image(
                    placeholder_img,
                    caption="Craniocaudal",
                    use_container_width=True
                )

        with mlo_image_holder:
            mlo_image_slot = st.empty()
            mlo_image_slot.image(
                placeholder_img,
                caption="Mediolateral Oblique",
                use_container_width=True
            )

    def build_footer(self):
        model_options, console, malignant_marker, benign_marker, foot_pad1 = st.columns([1.5, 6.5, 0.75, 0.75, 0.5])

        with model_options:
            st.write("Model Options"); st.button("⚙️")

        with console:
            st.write("Console"); st.code("sample console output")

        with malignant_marker:
            st.markdown(""); st.markdown("")
            st.button("Malignant", use_container_width=True, key="malignant_marker_button")

        with benign_marker:
            st.markdown(""); st.markdown("")
            st.button("Benign", use_container_width=True, key="benign_marker_button")

    def show_uploader(self):
        # dynamic key so widget can be created/destroyed cleanly
        key = f"dicom_uploader_{st.session_state.uploader_counter}"
        dicom_file = st.file_uploader(
            "Upload an image in DICOM format",
            type=["dcm"],
            key=key
        )

        if dicom_file is not None:
            # Convert UploadedFile -> pydicom Dataset
            dicom_ds = self.data_processor.uploadedfile_to_dicom(dicom_file)
            # Get display-ready uint8 pixels (your DP normalizes)
            pixels_uint8 = self.data_processor.extract_pixels_from_dicom(dicom_ds)

            # Persist and display
            st.session_state.cc_last_img = pixels_uint8
            self.cc_image_slot.image(pixels_uint8, caption="Craniocaudal", use_container_width=True)

            # Hide uploader and refresh so it disappears immediately
            st.session_state.show_uploader = False
            st.rerun()
