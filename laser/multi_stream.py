import cv2
import numpy as np
import traceback
import argparse
import pyrealsense2 as rs

parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream resolution, fps and format to match the recorded.")
parser.add_argument("--input", type=str, help="Path to the bag file")
parser.add_argument("--output", default='1.ply', type=str, help="Path to the bag file")
args = parser.parse_args()

def align():
    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        frames = pipeline.wait_for_frames()

        # 在对齐之前，先打印一遍信息
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        print(depth_image.shape)
        print(color_image.shape)

        # 对齐操作
        frames = align.process(frames)
        # 对齐之后，再打印一遍信息
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        print(depth_image.shape)
        print(color_image.shape)
    except:
        traceback.print_exc()
    finally:
        pipeline.stop()


def export_ply():
    pipe = rs.pipeline()
    config = rs.config()
    # 默认值是多少深度点？
    config.enable_stream(rs.stream.depth)
    pipe.start(config)

    # We'll use the colorizer to generate texture for our PLY
    # (alternatively, texture can be obtained from color or infrared stream)
    colorizer = rs.colorizer()

    try:
        frames = pipe.wait_for_frames()
        colorized = colorizer.process(frames)

        ply = rs.save_to_ply(args.output)
        ply.set_option(rs.save_to_ply.option_ply_binary, False)
        ply.set_option(rs.save_to_ply.option_ply_normals, True)

        ply.process(colorized)
        print("Done")
    except:
        traceback.print_exc()
    finally:
        pipe.stop()


def import_ply():
    pipeline = rs.pipeline()
    config = rs.config()
    # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, args.input)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    # Start streaming from file
    pipeline.start(config)

    # Create opencv window to render image in
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    # Create colorizer object
    colorizer = rs.colorizer()
    try:
        # Get frameset of depth
        frames = pipeline.wait_for_frames()

        # Get depth frame
        depth_frame = frames.get_depth_frame()

        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        # Render image in opencv window
        cv2.imshow("Depth Stream", depth_color_image)
        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
    except:
        traceback.print_exc()
    finally:
        pipeline.stop()




if __name__ == '__main__':
    main()