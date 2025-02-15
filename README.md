# Playing around with Intel RealSense D455 for researching CUDA acceleration with Computer Vision
I'm running on Linux so the following are the instructions for running:
# Setting Up
Make sure the necessary packages are available: </br>
`sudo apt install -y cmake build-essential libusb-1.0-0-dev libssl-dev libgtk-3-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev`
Install and build `librealsense` library from source: </br>
`git clone https://github.com/IntelRealSense/librealsense.git`
`cd librealsense`
`mkdir build`
`cd build`
`cmake .. -DBUILD_SHARED_LIBS=ON`
`make -j$(nproc)`
`sudo make install`
`export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH`
# Compile:
```g++ -o camera_stream camera_stream.cpp -I/usr/include/opencv4 -L/usr/local/lib -lrealsense2 $(pkg-config --cflags --libs opencv4)```
