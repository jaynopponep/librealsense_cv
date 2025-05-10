import streamlit as st
import os
import sys


'''
basically just logic to run  rgb capture and classifier, as well as pointcloud capture and classifier all in one page.
'''

#current working directory stuff
project_dir = os.getcwd()
sys.path.append(project_dir)

from RGB_capture_sign import capture_sign as rgb_capture_sign
from RGB_classifier import classifier as rgb_classifier
from PC_capture_sign import pc_capture_sign
from PC_classifier import classifier as pc_classifier

#configure streamlit page
st.set_page_config(page_title="RGB vs LiDAR ASL Classifier", layout="centered")

#title of page
st.title("RGB vs LiDAR ASL Classifier")

#choose your mode
mode = st.sidebar.selectbox(
    "Select Mode",
    ("Select Mode", "RGB Mode", "PointCloud Mode")
)
st.write(f"### Current Mode: {mode}")

if mode == "RGB Mode":
    if st.button("Capture ASL Sign (RGB)"):
        st.info("Starting RGB capture...")
        try:
            rgb_capture_sign()
            st.success("Capture complete! Landmarks saved to 2d_landmarks.parquet")
        except Exception as e:
            st.error(f"Capture failed: {e}")

    st.write("---")

    if st.button("Classify ASL Sign (RGB)"):
        st.info("Running RGB classifier...")
        try:
            sign = rgb_classifier()
            st.success(f"Predicted sign: **{sign}**")
        except Exception as e:
            st.error(f"Classification failed: {e}")

elif mode == "PointCloud Mode":
    if st.button("Capture ASL Sign (PointCloud)"):
        st.info("Starting PointCloud capture...")
        try:
            pc_capture_sign()
            st.success("Capture complete! Landmarks saved to 3d_landmarks.parquet")
        except Exception as e:
            st.error(f"Capture failed: {e}")

    st.write("---")

    if st.button("Classify ASL Sign (PointCloud)"):
        st.info("Running PointCloud classifier...")
        try:
            sign = pc_classifier()
            st.success(f"Predicted sign: **{sign}**")
        except Exception as e:
            st.error(f"Classification failed: {e}")

else:
    st.info("Please select a mode from the sidebar to begin.")

#footer
st.markdown(
    "_Use the sidebar to switch between RGB and PointCloud modes. Capture a short recording of a sign and classify it based on the saved landmarks._"
)