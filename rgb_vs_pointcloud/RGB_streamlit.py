import streamlit as st
import os
import sys

# Ensure the current working directory is on sys.path so we can import modules
project_dir = os.getcwd()
sys.path.append(project_dir)

from RGB_capture_sign import capture_sign
from RGB_classifier import classifier

# Configure Streamlit page
st.set_page_config(page_title="RGB ASL Classifier", layout="centered")

# Title
st.title("RGB ASL Classifier")

# Button to capture landmarks
if st.button("Capture ASL Sign"):
    st.info("Starting capture...")
    try:
        capture_sign()
        st.success("Capture complete! Landmarks saved to 2d_landmarks.parquet")
    except Exception as e:
        st.error(f"Capture failed: {e}")

st.write("---")

# Button to classify from the saved landmarks
if st.button("Classify ASL Sign"):
    st.info("Running classifier...")
    try:
        sign = classifier()
        st.success(f"Predicted sign: **{sign}**")
    except Exception as e:
        st.error(f"Classification failed: {e}")

# Footer note
st.markdown("_Use the buttons above to capture a short RGB recording of a sign and then classify it based on the saved landmarks._")