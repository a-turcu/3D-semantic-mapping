"""
!!! FILE CREATED FOR TESTING STUFF OUT !!!
"""



import numpy as np
from PIL import Image
from mmdet3d.apis import inference_detector, init_model

from utils_pcd import *
from const import *


def main():

    filename = "pointcloud_map_numpy"

    pcd = np.load(CLOUD_PATH + filename + ".npy")
    print(pcd.shape)
    #vis_o3d(pcd)

if __name__ == "__main__":
    main()