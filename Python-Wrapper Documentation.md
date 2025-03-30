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
- [ ] Git Clone the Point Net Repo for pointcloud examples
```bash
git clone https://github.com/charlesq34/pointnet.git
```
