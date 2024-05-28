#!/usr/bin/env python3
import rospy
import cv2
from cv_bridge import CvBridge, core
import imutils
import numpy as np
import pycuda.driver as drv
import pycuda.autoprimaryctx
from yoloDet import YoloTRT
from sensor_msgs.msg import Image 
from mavros_msgs.msg import PlayTuneV2, RCIn, GPSRAW, State
import rospkg
import os
import time
import pdb
import pyrealsense2 as rs


#need to explicitly mrention absol paths
LIBPLUGINS_PATH = os.path.join(rospkg.RosPack().get_path('JetsonYolov5'), 'yolov5', "build/libmyplugins.so")
TRTENGINE_PATH  = os.path.join(rospkg.RosPack().get_path('JetsonYolov5'), 'yolov5', "build/yolov5s.engine")


# use path for library and engine file
model = YoloTRT(library=LIBPLUGINS_PATH, engine=TRTENGINE_PATH, conf=0.5, yolo_ver="v5")

bridge = CvBridge()
ODT_Flag = False


cap = cv2.VideoCapture("videos/testvideo.mp4")

if __name__ =="__main__":
    Trial = False
    if Trial:
        while True:
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=600)
            detections, t = model.Inference(frame)
            # for obj in detections:
            #    print(obj['class'], obj['conf'], obj['box'])
            # print("FPS: {} sec".format(1/t))
            cv2.imshow("Output", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    
    else:

        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        pipeline.start(config)

        while (True):
            st = time.time()
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            detections, t = model.Inference(color_image)
            #   for obj in detections:
            #         print(obj['class'], obj['conf'], obj['box'])
            #   print("FPS: {} sec".format(1/t))
            cv2.imshow("Output", color_image)
            print(time.time() - st)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
