import numpy as np
from PIL import Image
from mmdet3d.apis import inference_detector, init_model

from utils_pcd import *
from const import *


def load_data(filename):
    depth_img = np.load(DEPTH_PATH + filename + ".npy")
    rgb_img = Image.open(RGB_PATH + filename + ".png")
    rgb_np = np.asarray(rgb_img, dtype=np.float32)
    info = load_pkl(INFO_PATH + filename + ".pkl")
    intrinsics = info['matrix_intrinsics']
    intrinsics = intrinsics.reshape(9)
    intrinsics = intrinsics.astype(np.float32)

    return depth_img, rgb_np, intrinsics


def analyze_result(result):
    """
    Extracts, thresholds and prints the result of the inference
    """
    predictions = result.pred_instances_3d
    scores_3d = predictions.scores_3d.cpu().numpy()
    bboxes_3d = predictions.bboxes_3d.tensor.cpu().numpy()
    labels_3d = predictions.labels_3d.cpu().numpy()

    indices = []
    for i, score in enumerate(scores_3d):
        if score > 0.5:
            indices.append(i)

    scores_3d = scores_3d[indices]
    bboxes_3d = bboxes_3d[indices]
    labels_3d = labels_3d[indices]

    # SUNRGBD classes
    # TODO move somewhere else
    classes = [
        'bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
        'night_stand', 'bookshelf', 'bathtub'
    ]

    for i in labels_3d:
        print(classes[i])


def main():

    filename = "000016"
    pcd_path = CLOUD_PATH + filename + ".npy"

    depth_img, rgb_np, intrinsics = load_data(filename)
    pcd = create_pcd(depth_img, rgb_np, intrinsics, filename)

    model = init_model(CONFIG_FILE, CHCKPOINT_FILE)
    result, data = inference_detector(model, pcd_path)

    analyze_result(result)
    vis_mm(model, data, result)

if __name__ == "__main__":
    main()