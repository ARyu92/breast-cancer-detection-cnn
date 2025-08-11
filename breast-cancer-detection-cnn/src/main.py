import streamlit as st
from ui.GUI import GUI
from model.model import Model
import os
from pathlib import Path



class MammogramApplication:
    def __init__(self):
        
        self.gui = GUI()

        cnn_model = Model()
        model_path = cnn_model.TRAINED_MODELS_DIR / "default" / "default.keras"

        if model_path.exists():
            cnn_model.load_model(model_path)
        else:
            st.error(f"Model not found at {model_path}")

    def main(self):
       
        return

if __name__ == "__main__":
    MammogramApplication().main()