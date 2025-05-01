## Installation Instruction
- [ ] Install python 3.7.2  **MUST BE 64 BIT VERSION NOT 32**
```
https://www.python.org/downloads/release/python-372/
```
- [ ] Create a 3.7.2 venv
```bash
py -3.7 -m venv .venv
.venv/Scripts/activate
```
or on Linux and mac:
```bash
source .venv/Scripts/activate
```

- [ ] Pip install libraries
```bash
pip install -r requirements.txt
```

- [ ] Be sure to CD into the rgb vs pointcloud directory
```bash 
cd rgb_vs_pointcloud
```

## Running the RGB Model
- [ ] First run the RGB_capture_sign.py file
```bash
py RGB_capture_sign.py
```
You should see a map open. This is the pointcloud captured from the lidar camera, converted into a collection of np arrays. It is rendered in Open3D for 3-D viewing
- [ ] Now run the PC_classifier.py file
```bash
py PC_classifier.py
```
- [ ] To run and see both models at the same time, run the following:
```bash
py RGB_streamlit.py
```
## Running the Lidar/Pointcloud Model
- [ ] First run the PC_capture_sign.py file
```bash
py PC_capture_sign.py
```
You should see a map open. This is the pointcloud captured from the lidar camera, converted into a collection of np arrays. It is rendered in Open3D for 3-D viewing
- [ ] Now run the PC_classifier.py file
```bash
py PC_classifier.py
```
This will run the Kaggle model against the captured pointclouds, recently turned into collection of np arrays. 

- [ ] Run the PointCloud Streamlit. Do note you must specify a different port for this streamlit to be running at the same time as the RGB streamlit
```bash
streamlit run PC_streamlit.py --server.port 8502
```




