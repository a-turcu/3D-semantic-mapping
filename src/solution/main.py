import numpy as np
import json
from mmdet3d.apis import init_model

from const import *
from utils_pcd import open3d_to_numpy, transform_pcd
from registration import iterative_global_registration
from detection import detect, create_result


def main():

    save_path = "live_pcd"

    # perform detection on global point cloud
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE)
    result = detect(model, save_path + ".npy")
    print("ok")

if __name__ == "__main__":
    main()