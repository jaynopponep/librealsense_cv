import streamlit as st
from PC_capture_sign import pc_capture_sign   #importing the pointcloud capture and classifier functions
from PC_classifier import classifier       

#it is extremely important to specify a different port for the pointcloud streamlit app, as it will conflict with the RGB streamlit app otherwise.
#please run the following command in the terminal to run the streamlit app:
# streamlit run PC_streamlit.py --server.port 8502

st.set_page_config(page_title="PointCloud ASL Classifier", layout="centered")
st.title("PointCloud ASL Classifier")

if st.button("Capture ASL Sign (PointCloud)"):
    st.info("Starting pointcloud capture…")
    try:
        pc_capture_sign()
        st.success("Capture complete! Landmarks saved to 3d_landmarks.parquet")
    except Exception as e:
        st.error(f"Capture failed: {e}")

st.write("---")

if st.button("Classify ASL Sign (PointCloud)"):
    st.info("Running classifier…")
    try:
        sign = classifier()
        st.success(f"Predicted sign: **{sign}**")
    except Exception as e:
        st.error(f"Classification failed: {e}")

st.markdown(
    "_Use the buttons above to capture a short point‑cloud recording of a sign and then classify it based on the saved landmarks._"
)
