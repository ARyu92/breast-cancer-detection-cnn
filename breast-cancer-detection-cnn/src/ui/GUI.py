import streamlit as st
import numpy as np

class GUI:
    def __init__(self):
        st.set_page_config(layout="wide")
        #Header
        #st.title("Breast Cancer Detection")
        #st.markdown("An AI Powered Application")

        #Header
        head_pad1, head_col1, head_pad2,  = st.columns([1.5, 1, 7.5])

        with head_col1:
            st.button("Load Image", use_container_width= True, )

        #Main body
        placeholder_img = np.zeros((400, 400, 3), dtype=np.uint8)

        col1, col2, col3, col_pad1 = st.columns([1.5,4,4,0.5])

        with col1: 
            st.subheader("Patient:")
            name_box = st.empty()
            name = "Dave"

            
            name_box.write(f"Patient Name: {name}")

        with col2:
            #st.subheader("CC Image")
            st.image(placeholder_img, caption="Craniocaudal", use_container_width =True)

        with col3:
            #st.subheader("MLO Image")
            st.image(placeholder_img, caption= "Mediolateral Oblique", use_container_width =True)

        


        #Footer 
        foot_col1, foot_col2, foot_col3, foot_col4, foot_pad1 = st.columns([1.5, 6.5, 0.75 ,0.75, 0.5])

        with foot_col1:
            st.write("Model")
            st.button("⚙️")

        with foot_col2:
            output_text = ""
            st.write("Console")
            st.markdown(f"""
                <div style='background-color: white; padding: 20px; height: 100px; border-radius: 8px;
                            box-shadow: 0 0 5px rgba(20,20,20,0.1); overflow-y: auto; white-space: pre-line;'>
                    <p>{output_text}</p>
                </div>
            """, unsafe_allow_html=True)

        with foot_col3:
            st.markdown("<br>", unsafe_allow_html= True)
            st.button("Malignant", key = "malignant")

        with foot_col4:
            st.markdown("<br>", unsafe_allow_html= True)
            st.button("Benign", key = "benign")

