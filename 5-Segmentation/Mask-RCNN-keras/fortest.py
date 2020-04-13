# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
from mrcnn.config import Config
from datetime import datetime

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
 
COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"mask_rcnn_shapes_0014.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("------------------")
 
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
 
class ShapesConfig(Config):
    NAME = "shapes"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 7  # 1bg + 7 shapes

    IMAGE_MIN_DIM = 1088
    IMAGE_MAX_DIM = 1920
 
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    TRAIN_ROIS_PER_IMAGE =100
 
    STEPS_PER_EPOCH = 50
    VALIDATION_STEPS = 50


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['car', 'truck', 'van', 'bus', 'cycle', 'tricycle', 'person']
image = skimage.io.imread(".\\standard_apollo_mask_datasets\\test\\000015.png")
 
a = datetime.now()
results = model.detect([image], verbose=1)
b = datetime.now()
print("cost: ", (b - a).seconds)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
 

