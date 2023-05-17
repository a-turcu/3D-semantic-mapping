import numpy as np
import torch
from PIL import Image
from mmdet3d.apis import inference_detector, init_model
import os

from const import *
from mmdet3d.structures import DepthInstance3DBoxes
from mmdet3d.visualization import Det3DLocalVisualizer
from utils_pcd import create_pcd, load_pkl, vis_mm, random_sampling
import open3d as o3d
from benchbot_eval.ground_truth_creator import GroundTruthCreator


def load_data(filename):
    depth_np = np.load(DEPTH_PATH + filename + ".npy")
    rgb_img = Image.open(RGB_PATH + filename + ".png")
    rgb_np = np.asarray(rgb_img, dtype=np.float32)
    info = load_pkl(INFO_PATH + filename + ".pkl")
    intrinsics = info['matrix_intrinsics']
    intrinsics = intrinsics.reshape(9)
    intrinsics = intrinsics.astype(np.float32)
    poses = load_pkl(POSE_PATH + filename + ".pkl")

    return depth_np, rgb_np, intrinsics, poses


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

    #challenge_classes = classes_sunrgbd[:6]
    #challenge_classes[5] = 'table'

    for index, val in enumerate(labels_3d):
        print(classes_sunrgbd[val], bboxes_3d[index])
        print('\n')


def generate_all_pcds():

    for i in range(len(os.listdir(DEPTH_PATH_PASSIVE))):
        if i < 10:
            filename = "00000" + str(i)
        else:
            filename = "0000" + str(i)

        depth_img, rgb_np, intrinsics, poses = load_data(filename)
        create_pcd(depth_img, rgb_np, intrinsics, filename)


def vis_mm_2(pcd, color=False, boxes=None):
    visualizer = Det3DLocalVisualizer(pcd_mode=2)
    if color:
        visualizer.set_points(pcd, mode='xyzrgb')
    else:
        visualizer.set_points(pcd, mode='xyz')
    if boxes is not None:
        visualizer.draw_bboxes_3d(boxes)
    visualizer.show()


def transform_pcd(pcd):

    # sample pcd for faster testing
    pcd = random_sampling(pcd, 500_000)

    # numpy to o3d
    points = pcd[:, :3] / 1000
    colors = pcd[:, 3:6] / 255.0
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(points)
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors)

    # transform
    R = pcd_o3d.get_rotation_matrix_from_xyz((0, 0, 0.63*np.pi))
    pcd_o3d = pcd_o3d.rotate(R, center=(0, 0, 0))
    pcd_o3d = pcd_o3d.translate((0.0011, 0.0012, 0.0012))
    #pcd_o3d = pcd_o3d.scale(1, center=(0, 0, 0))

    # o3d to numpy
    pcd = np.asarray(pcd_o3d.points)
    rgb_np = np.asarray(pcd_o3d.colors)
    pcd = np.concatenate((pcd, rgb_np.reshape(-1, 3)), axis=1)
    pcd = pcd.astype(np.float32)
    pcd[:, :3] = pcd[:, :3] * 1000

    return pcd


def main():

    filename = "global_pcd"

    pcd_path = CLOUD_PATH + filename + ".npy"
    pcd = np.load(pcd_path)

    pcd = transform_pcd(pcd)

    transformed_path = CLOUD_PATH + "global_pcd_transformed.npy"
    np.save(transformed_path, pcd)

    threshold = 0.7
    model = init_model(CONFIG_FILE, CHCKPOINT_FILE)
    result, data = inference_detector(model, transformed_path)

    # bed
    box_bed = DepthInstance3DBoxes(torch.tensor([[2.20074341, 0.05944466, 0.44632973, 2.29851074, 1.85151764, 0.89577698, 0]]))

    # sink
    box_sink = DepthInstance3DBoxes(torch.tensor([[-0.58994232, 3.07765717, 0.82831787, 0.32866461999999996, 0.27531434, 0.14530166, 0]]))

    # table 1
    box_table1 = DepthInstance3DBoxes(torch.tensor([[0.47003494, -0.53999237, 0.17274715000000002, 0.60470368, 0.40477172000000006, 0.34549430000000003, 0]]))

    # table 2
    box_table2 = DepthInstance3DBoxes(torch.tensor([[3.11474518, 1.08274681, 0.17274715000000002, 0.40477112, 0.60470322, 0.34549430000000003, 0]]))

    # center
    box_center = DepthInstance3DBoxes(torch.tensor([[-0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0]]))

    # result.pred_instances_3d.bboxes_3d.tensor[0, :] = box_bed.tensor
    # result.pred_instances_3d.bboxes_3d.tensor[1, :] = box_sink.tensor
    # result.pred_instances_3d.bboxes_3d.tensor[2, :] = box_table1.tensor
    # result.pred_instances_3d.bboxes_3d.tensor[3, :] = box_table2.tensor
    # result.pred_instances_3d.bboxes_3d.tensor[4:, :] = box_center.tensor
    #
    # result.pred_instances_3d.scores_3d[0] = 0.99
    # result.pred_instances_3d.scores_3d[1] = 0.99
    # result.pred_instances_3d.scores_3d[2] = 0.99
    # result.pred_instances_3d.scores_3d[3] = 0.99


    vis_mm(model, data, result, threshold)
    analyze_result(result, threshold)
    print("ok")


if __name__ == "__main__":
    main()