## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##                  Export to PLY                  ##
#####################################################

# First import the library
import numpy as np
import pyrealsense2 as rs


# # Declare pointcloud object, for calculating pointclouds and texture mappings
# pc = rs.pointcloud()
# # We want the points object to be persistent so we can display the last cloud when a frame drops
# points = rs.points()
#
# # Declare RealSense pipeline, encapsulating the actual device and sensors
# pipe = rs.pipeline()
# config = rs.config()
# # Enable depth stream
# config.enable_stream(rs.stream.depth)
# # config.enable_stream(rs.stream.color)
#
# # Start streaming with chosen configuration
# pipe.start(config)
#
# # We'll use the colorizer to generate texture for our PLY
# # (alternatively, texture can be obtained from color or infrared stream)
# colorizer = rs.colorizer()





pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)
# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
try:
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    '''
    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    print(depth_image.shape, color_image.shape)
    '''

    # Create save_to_ply object
    ply = rs.save_to_ply("1.ply")

    # Set options to the desired values
    # In this example we'll generate a textual PLY with normals (mesh is already created by default)
    ply.set_option(rs.save_to_ply.option_ply_binary, True)
    ply.set_option(rs.save_to_ply.option_ply_normals, False)

    print("Saving to 1.ply...")
    # Apply the processing block to the frameset which contains the depth frame and the texture
    # ply.process(colorized)
    # ply.process(color_frame)
    # ply.process(aligned_depth_frame)
    ply.process(aligned_frames)
    print("Done")

finally:
    pipeline.stop()
