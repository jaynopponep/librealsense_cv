import pyrealsense2 as rs
#this is just a debug script to see if l515 is detected. should say camera detected: ['Intel RealSense L515']
#make sure ur using a usb c 3.0 cable, <3.0 connections not allowed for l515
context = rs.context()
devices = context.query_devices()

if devices:
    print("Camera detected:", [device.get_info(rs.camera_info.name) for device in devices])
else:
    print("No RealSense camera found.")
