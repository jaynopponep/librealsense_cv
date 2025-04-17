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
pip install pyrealsense2
python -m pip install --upgrade pip
pip install numpy opencv-python pandas open3d tensorflow pyarrow pyglet==1.4.9
pip install "urllib3<2"
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

- [ ] You can also view both in the streamlit file
```bash
py PC_streamlit.py
```
## Running the RGB Model

- [ ] Create a new venv for python 3.9+
```bash
py -3.12 venv .venv2
.venv2/Scripts/activate
```
or on mac/linux
```bash
source .venv/Scripts/activate
```
- [ ] Pip install requirements
```bash
pip install requirements3-12.txt
```
- [ ] Run the RGB Streamlit
```bash
py RGB_streamlit.py
```
