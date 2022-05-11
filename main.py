import cv2

import tensorflow as tf

import numpy as np
import os

from my_yolov4.tf import YOLOv4

if __name__ == "__main__":
    yolo = YOLOv4()
    yolo.config.parse_names("test/dataset/coco.names")
    yolo.config.parse_cfg("config/yolov4-tiny.cfg")

    yolo.make_model()
    yolo.load_weights("yolov4-tiny.weights", weights_type="yolo")
    yolo.summary(summary_type="yolo")
    yolo.summary()

    # change the media path below to see the result for image
    yolo.inference(media_path="cars/data/train/police_car/003_e56e8351.jpg")

    # Comment out below to see the result for video
    # yolo.inference(
    #     "police_car_video.mp4",
    #     is_image=False,
    #     cv_apiPreference=cv2.CAP_V4L2,
    #     cv_frame_size=(640, 480),
    #     cv_fourcc="YUYV",
    # )