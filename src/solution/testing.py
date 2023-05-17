"""
!!! FILE CREATED FOR TESTING STUFF OUT !!!
"""
import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from mmdet3d.visualization.local_visualizer import Det3DLocalVisualizer
from mmdet3d.registry import VISUALIZERS
from mmdet3d.apis import inference_detector, init_model
#from mmdet.apis import init_detector, inference_detector
#from mmdet.utils import register_all_modules
#from mmdet.registry import VISUALIZERS
import mmcv
from main import load_data, vis_mm
from utils_pcd import load_pkl, create_pcd, random_sampling
from const import *


def vis_mm_2(pcd, color=False, boxes=None):
    visualizer = Det3DLocalVisualizer(pcd_mode=2)
    if color:
        visualizer.set_points(pcd, mode='xyzrgb')
    else:
        visualizer.set_points(pcd, mode='xyz')
    if boxes is not None:
        visualizer.draw_bboxes_3d(boxes)
    visualizer.show()


def compute_world_T_camera(pose_rotation, pose_translation):
    """
    This function computes the transformation matrix T that transforms one point in the camera coordinate system to
    the world coordinate system. This function needs the scipy package - scipy.spatial.transform.Rotation
    P_w = P_r * T, where P_r = [x, y, z] is a row vector in the camera coordinate (x - right, y - down, z - forward)
    and P_w the location of the point in the world coordinate system (x - right, y - forward, z - up)

    :param pose_rotation: [quat_x, quat_y, quat_z, quat_w], a quaternion that represents a rotation
    :param pose_translation: [tr_x, tr_y, tr_z], the location of the robot origin in the world coordinate system
    :return: transformation matrix
    """
    # compute the rotation matrix
    # note that the scipy...Rotation accepts the quaternion in scalar-last format
    pose_quat = pose_rotation
    rot = Rotation.from_quat(pose_quat)
    rotation_matrix = np.eye(4)
    rotation_matrix[0:3][:, 0:3] = rot.as_dcm().T

    # build the translation matrix
    translation_matrix = np.eye(4)
    translation_matrix[3][0:3] = np.array(pose_translation)

    # compute the transformation matrix
    transformation_matrix = rotation_matrix @ translation_matrix

    # # convert robot's right-handed coordinate system to unreal world's left-handed coordinate system
    #transformation_matrix[:, 1] = -1.0 * transformation_matrix[:, 1]

    return transformation_matrix


def convert_box_to_world(box_corners_camera, w_T_c):

    box_corners_camera = np.concatenate(
        (box_corners_camera,
         np.ones((box_corners_camera.shape[0], 1),
                 dtype=box_corners_camera.dtype)),
        axis=1)

    box_corners_world = box_corners_camera @ w_T_c
    box_corners_world = box_corners_world[:, 0:3]

    # compute the 3D box's center roughly
    box_center_world = np.sum(box_corners_world, axis=0) / 8.0

    # compute the 3D box's extent roughly
    box_extent_world = np.sum(
        np.abs(box_corners_world - box_center_world), axis=0) / 4.0


    return box_corners_world, box_center_world, box_extent_world


def analyze_result(result, w_T_c, threshold=0):
    """
    Extracts, thresholds and prints the result of the inference
    """

    predictions = result.pred_instances_3d
    scores_3d = predictions.scores_3d.cpu().numpy()
    bboxes_3d = predictions.bboxes_3d
    #box_centroids = bboxes_3d.corners.cpu().numpy()
    labels_3d = predictions.labels_3d.cpu().numpy()

    indices = []
    for i, score in enumerate(scores_3d):
        if score > threshold and labels_3d[i] < 6:
            indices.append(i)

    scores_3d = scores_3d[indices]
    bboxes_3d = bboxes_3d[indices]
    labels_3d = labels_3d[indices]

    box_corners = bboxes_3d.corners.cpu().numpy()

    # SUNRGBD classes
    # TODO move somewhere else
    classes_sunrgbd = [
        'bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
        'night_stand', 'bookshelf', 'bathtub'
    ]

    #challenge_classes = classes_sunrgbd[:6]
    #challenge_classes[5] = 'table'


    frame_results = []
    for index, val in enumerate(labels_3d):
        #transformed_bbox = convert_box_to_world(box_centroids[index], w_T_c)
        #print(classes_sunrgbd[val])
        corners, center, extent = convert_box_to_world(box_corners[index], w_T_c)
        result = {
            "class": classes_sunrgbd[labels_3d[index]],
            "class_ID": labels_3d[index],
            "confidence": scores_3d[index],
            "box_corners_camera": box_corners[index],
            "box_corners_world": corners,
            "centroid": center,
            "extent": extent
        }
        frame_results.append(result)

    return frame_results


def votenet_nms(all_results, class_list):
    nms_iou_cls = 0.1  # this threshold for removing duplicates that are of the same class
    boxes_world = []
    results_list = []
    for results in all_results:
        if not results:
            continue
        for result in results:
            box_world = np.zeros((1, 8), dtype=np.float32)
            box_corners_world = result["box_corners_world"]
            box_world[0, 0] = np.min(box_corners_world[:, 0])
            box_world[0, 1] = np.min(box_corners_world[:, 1])
            box_world[0, 2] = np.min(box_corners_world[:, 2])
            box_world[0, 3] = np.max(box_corners_world[:, 0])
            box_world[0, 4] = np.max(box_corners_world[:, 1])
            box_world[0, 5] = np.max(box_corners_world[:, 2])
            box_world[0, 6] = result["confidence"]
            box_world[0, 7] = result["class_ID"]
            boxes_world.append(box_world)
            results_list.append(result)
    boxes_world = np.concatenate(boxes_world)

    pick = nms_3d_faster_samecls(boxes_world, nms_iou_cls)
    assert (len(pick) > 0)
    results_intermediate = [results_list[p] for p in pick]

    nms_iou = 0.25  # this threshold is used to remove duplicates that are not of the same class
    boxes_world = []
    results_list = []
    for result in results_intermediate:
        if not result:
            continue
        box_world = np.zeros((1, 7), dtype=np.float32)
        box_corners_world = result["box_corners_world"]
        box_world[0, 0] = np.min(box_corners_world[:, 0])
        box_world[0, 1] = np.min(box_corners_world[:, 1])
        box_world[0, 2] = np.min(box_corners_world[:, 2])
        box_world[0, 3] = np.max(box_corners_world[:, 0])
        box_world[0, 4] = np.max(box_corners_world[:, 1])
        box_world[0, 5] = np.max(box_corners_world[:, 2])
        box_world[0, 6] = result["confidence"]
        # box_world[0, 7] = result["class_ID"]
        boxes_world.append(box_world)
        results_list.append(result)
    boxes_world = np.concatenate(boxes_world)

    pick = nms_3d_faster(boxes_world, nms_iou)
    assert (len(pick) > 0)

    results_final = [results_list[p] for p in pick]
    results_final = jsonify(results_final)

    for r in results_final:
        r['label_probs'] = [0] * len(class_list)
        if r['class'] in class_list:
            r['label_probs'][class_list.index(r['class'])] = r['confidence']

    return results_final


def jsonify(data_list):
    json_list = []
    for data in data_list:
        json_data = dict()
        for key, value in data.items():
            if isinstance(value, list):  # for lists
                value = [
                    jsonify(item) if isinstance(item, dict) else item
                    for item in value
                ]
            if isinstance(value, dict):  # for nested lists
                value = jsonify(value)
            if isinstance(key, int):  # if key is integer: > to string
                key = str(key)
            if type(
                    value
            ).__module__ == 'numpy':  # if value is numpy.*: > to python list
                value = value.tolist()
            json_data[key] = value
        json_list.append(json_data)
    return json_list


def nms_3d_faster(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    z1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]
    z2 = boxes[:,5]
    score = boxes[:,6]
    area = (x2-x1)*(y2-y1)*(z2-z1)

    I = np.argsort(score)
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[:last-1]])
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        zz1 = np.maximum(z1[i], z1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])
        zz2 = np.minimum(z2[i], z2[I[:last-1]])

        l = np.maximum(0, xx2-xx1)
        w = np.maximum(0, yy2-yy1)
        h = np.maximum(0, zz2-zz1)

        if old_type:
            o = (l*w*h)/area[I[:last-1]]
        else:
            inter = l*w*h
            o = inter / (area[i] + area[I[:last-1]] - inter)

        I = np.delete(I, np.concatenate(([last-1], np.where(o>overlap_threshold)[0])))

    return pick


def nms_3d_faster_samecls(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    z1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]
    z2 = boxes[:,5]
    score = boxes[:,6]
    cls = boxes[:,7]
    area = (x2-x1)*(y2-y1)*(z2-z1)

    I = np.argsort(score)
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[:last-1]])
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        zz1 = np.maximum(z1[i], z1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])
        zz2 = np.minimum(z2[i], z2[I[:last-1]])
        cls1 = cls[i]
        cls2 = cls[I[:last-1]]

        l = np.maximum(0, xx2-xx1)
        w = np.maximum(0, yy2-yy1)
        h = np.maximum(0, zz2-zz1)

        if old_type:
            o = (l*w*h)/area[I[:last-1]]
        else:
            inter = l*w*h
            o = inter / (area[i] + area[I[:last-1]] - inter)
        o = o * (cls1==cls2)

        I = np.delete(I, np.concatenate(([last-1], np.where(o>overlap_threshold)[0])))

    return pick

def main():

    classes_sunrgbd = [
        'bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
        'night_stand', 'bookshelf', 'bathtub'
    ]
    threshold = 0.9
    all_results = []
    model = init_model(CONFIG_FILE, CHCKPOINT_FILE)

    for i in range(87):

        filename = "0" * (6 - len(str(i))) + str(i)
        print(f"Loading data: {filename}")
        pcd_path = CLOUD_PATH + filename + ".npy"
        depth_img, rgb_np, intrinsics, poses = load_data(filename)
        pcd = create_pcd(depth_img, rgb_np, intrinsics, filename)


        result, data = inference_detector(model, pcd_path)


        #vis_mm(model, data, result, threshold)

        camera_pose = poses['camera']
        w_T_c = compute_world_T_camera(camera_pose['rotation_xyzw'], camera_pose['translation_xyz'])

        all_results.append(analyze_result(result, w_T_c, threshold))

    results = votenet_nms(all_results, classes_sunrgbd)
    print("ok")

if __name__ == "__main__":
    main()
