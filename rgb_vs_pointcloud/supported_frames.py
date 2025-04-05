import pyrealsense2 as rs

ctx = rs.context()
device = ctx.query_devices()[0]
sensor = device.query_sensors()[0]  # depth sensor is usually the first

print("Supported depth modes:")
for profile in sensor.get_stream_profiles():
    if profile.stream_type() == rs.stream.depth:
        vsp = profile.as_video_stream_profile()
        print(f" - {vsp.width()}x{vsp.height()} @ {profile.fps()}Hz, format={profile.format()}")
