import numpy as np
from PIL import Image
from mmdet3d.apis import inference_detector, init_model
import os

from const import *
from utils_pcd import create_pcd, load_pkl, vis_mm
from iterative_global_registration import create_global_pointcloud

def load_data(filename):
    depth_img = np.load(DEPTH_PATH + filename + ".npy")
    rgb_img = Image.open(RGB_PATH  + filename + ".png")
    rgb_np = np.asarray(rgb_img, dtype=np.float32)
    info = load_pkl(INFO_PATH  + filename + ".pkl")
    intrinsics = info['matrix_intrinsics']
    intrinsics = intrinsics.reshape(9)
    intrinsics = intrinsics.astype(np.float32)
    poses = load_pkl(POSE_PATH + filename + ".pkl")

    return depth_img, rgb_np, intrinsics, poses


def analyze_result(result, threshold=0):
    """
    Extracts, thresholds and prints the result of the inference
    """
    predictions = result.pred_instances_3d
    scores_3d = predictions.scores_3d.cpu().numpy()
    bboxes_3d = predictions.bboxes_3d.tensor.cpu().numpy()
    labels_3d = predictions.labels_3d.cpu().numpy()

    indices = []
    for i, score in enumerate(scores_3d):
        if score > threshold:
            indices.append(i)

    scores_3d = scores_3d[indices]
    bboxes_3d = bboxes_3d[indices]
    labels_3d = labels_3d[indices]

    # SUNRGBD classes
    # TODO move somewhere else
    classes_sunrgbd = [
        'bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
        'night_stand', 'bookshelf', 'bathtub'
    ]

    challenge_classes = classes_sunrgbd[:6]
    challenge_classes[5] = 'table'

    for i in labels_3d:
        print(classes_sunrgbd[i])



def generate_all_pcds():

    for i in range(len(os.listdir(DEPTH_PATH_PASSIVE))):
        if i < 10:
            filename = "00000" + str(i)
        else:
            filename = "0000" + str(i)

        depth_img, rgb_np, intrinsics, poses = load_data(filename)
        create_pcd(depth_img, rgb_np, intrinsics, filename)


def main():

    filename = "global_pcd"
    pcd_path = CLOUD_PATH + filename + ".npy"
    if not os.path.exists(pcd_path):
        create_global_pointcloud(pcd_path)

    threshold = 0.7
    model = init_model(CONFIG_FILE, CHCKPOINT_FILE)
    result, data = inference_detector(model, pcd_path)

    analyze_result(result, threshold)
    vis_mm(model, data, result, threshold)


if __name__ == "__main__":
    main()