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
- [ ] Pip install libraries
```bash
pip install pyrealsense2
python -m pip install --upgrade pip
pip install numpy opencv-python pandas open3d tensorflow pyarrow pyglet==1.4.9
pip install "urllib3<2"
```

## Running the Model
- [ ] First run the pc3d.py file
```bash
py pc3d.py
```
You should see a map open. This is the pointcloud captured from the lidar camera, converted into a collection of np arrays. It is rendered in Open3D for 3-D travel
- [ ] Now run the classifier.py file
```bash
py classifier.py
```
This will run the Kaggle model against the captured pointclouds, recently turned into collection of np arrays. 
