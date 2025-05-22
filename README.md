## Installation Instruction
- [ ] Install python 3.12  **MUST BE 64 BIT VERSION NOT 32**
```
https://www.python.org/downloads/release/python-31210/
```
- [ ] Create a 3.12 venv
```bash
python -3.12 -m venv .venv
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
python RGB_capture_sign.py
```
You should see a map open. This is the pointcloud captured from the lidar camera, converted into a collection of np arrays. It is rendered in Open3D for 3-D viewing
- [ ] Now run the PC_classifier.py file
```bash
python PC_classifier.py
```



## Running the Lidar/Pointcloud Model
- [ ] First run the PC_capture_sign.py file
```bash
python PC_capture_sign.py
```
You should see a map open. This is the pointcloud captured from the lidar camera, converted into a collection of np arrays. It is rendered in Open3D for 3-D viewing
- [ ] Now run the PC_classifier.py file
```bash
python PC_classifier.py
```
This will run the Kaggle model against the captured pointclouds, recently turned into collection of np arrays. 




## Run Both RGB and Lidar
- [ ] Run the  Streamlit. This allows you to run and see both models at the same time
```bash
streamlit run streamlit_asl.py
```
