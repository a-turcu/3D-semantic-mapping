"""
!!! FILE CREATED FOR TESTING STUFF OUT !!!
"""
import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from mmdet3d.visualization.local_visualizer import Det3DLocalVisualizer
from detection import detect, vis_mm, vis_mm_2
from const import *
from mmdet3d.apis import init_model, inference_detector


def main():


    pcd = np.load("pcd.npy")
    model = init_model(CONFIG_FILE, CHCKPOINT_FILE)

    # split pcd in


if __name__ == "__main__":
    main()
