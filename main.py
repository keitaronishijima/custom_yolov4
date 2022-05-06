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

    # change the media path below to see the result
    
    
    yolo.inference(media_path="ambulance7.jpeg")
    # train_path = "car/data/train"
    # validation_path = "car/data/val"
    # test_path  = "car/data/test"


    # categories = os.listdir(train_path)
    # categories.sort()
    # IMG_SIZE = 100
    # img = 'car/data/train/pickup_truck/0DL5XXBD9R5B.jpg'
    # #img = 'food/train/banana/Image_9.jpg'
    # img = cv2.imread(img)
    # img_arr = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    # reconstructed_model = tf.keras.models.load_model("ambulance_model.h5")
    # img_arr = np.expand_dims(img_arr, axis = 0)
    # print(img_arr.shape)
    # res = np.squeeze(reconstructed_model.predict(img_arr))
    # idx = 0
    # for i in range(len(res)):
    #     if res[i] == 1:
    #         print(categories[i])
    

    # #Comment out below to see the result for video
    # yolo.inference(
    #     "ambulance_video.mov",
    #     is_image=False,
    #     cv_apiPreference=cv2.CAP_V4L2,
    #     cv_frame_size=(640, 480),
    #     cv_fourcc="YUYV",
    # )