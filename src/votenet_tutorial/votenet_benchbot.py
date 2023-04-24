# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This is a application of Votenet to the BenchBot Semantic SLAM problem

import os
import sys
import numpy as np
import argparse
import importlib
import time
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.spatial.transform import Rotation

# Import votenet helpers explicitly (as the package does not offer any pip
# install tools)
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'votenet')
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pc_util import random_sampling, read_ply
from ap_helper import parse_predictions
from nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls

# Default settings
DATASET = 'sunrgbd'  # supported values: sunrgbd, scannet
NUM_POINTS = 40000  # number of points used in sampling the point cloud


def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:, 0:3]  # do not use color for now
    floor_height = np.percentile(point_cloud[:, 2], 0.99)
    height = point_cloud[:, 2] - floor_height
    point_cloud = np.concatenate(
        [point_cloud, np.expand_dims(height, 1)], 1)  # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, NUM_POINTS)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0)  # (1,40000,4)
    return pc


def votenet_build():
    # Set file paths and dataset config
    demo_dir = os.path.join(ROOT_DIR, 'demo_files')
    if DATASET == 'sunrgbd':
        sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
        from sunrgbd_detection_dataset import DC  # dataset config

        checkpoint_path = os.path.join(demo_dir,
                                       'pretrained_votenet_on_sunrgbd.tar')
        pc_path = os.path.join(demo_dir, 'input_pc_sunrgbd.ply')
    elif DATASET == 'scannet':
        sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
        from scannet_detection_dataset import DC  # dataset config

        checkpoint_path = os.path.join(demo_dir,
                                       'pretrained_votenet_on_scannet.tar')
        pc_path = os.path.join(demo_dir, 'input_pc_scannet.ply')
    else:
        print('Unkown dataset %s. Exiting.' % (DATASET))
        exit(-1)

    eval_config_dict = {
        'remove_empty_box': True,
        'use_3d_nms': True,
        'nms_iou': 0.25,
        'use_old_type_nms': False,
        'cls_nms': False,
        'per_class_proposal': False,
        'conf_thresh': 0.5,
        'dataset_config': DC
    }

    # Init the model and optimzier
    MODEL = importlib.import_module('votenet')  # import network module
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = MODEL.VoteNet(num_proposal=256,
                        input_feature_dim=1,
                        vote_factor=1,
                        sampling='seed_fps',
                        num_class=DC.num_class,
                        num_heading_bin=DC.num_heading_bin,
                        num_size_cluster=DC.num_size_cluster,
                        mean_size_arr=DC.mean_size_arr).to(device)
    print('Constructed model.')

    # Load checkpoint
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("Loaded checkpoint %s (epoch: %d)" % (checkpoint_path, epoch))

    net.device = device
    net.eval_config_dict = eval_config_dict

    type2class = {
        'bed': 0,
        'table': 1,
        'sofa': 2,
        'chair': 3,
        'toilet': 4,
        'desk': 5,
        'dresser': 6,
        'night_stand': 7,
        'bookshelf': 8,
        'bathtub': 9
    }
    class2type = {type2class[t]: t for t in type2class}
    # merge 'table:1' and 'desk:5' into 'table'
    class2type[5] = 'table'
    net.type2class = type2class
    net.class2type = class2type

    return net


def votenet_forward(net, point_cloud):
    # Load and preprocess input point cloud
    net.eval()  # set model to eval mode (for bn and dp)
    # point_cloud = read_ply(pc_path)
    pc = preprocess_point_cloud(point_cloud)
    # print('Loaded point cloud data: %s' % (pc_path))

    # Model inference
    inputs = {'point_clouds': torch.from_numpy(pc).to(net.device)}
    tic = time.time()
    with torch.no_grad():
        end_points = net(inputs)
    toc = time.time()
    print('Inference time: %f' % (toc - tic))
    end_points['point_clouds'] = inputs['point_clouds']
    try:
        pred_map_cls = parse_predictions(end_points, net.eval_config_dict)
        print('Finished detection. %d object detected.' %
              (len(pred_map_cls[0])))
        return pred_map_cls
    except:
        print('Finished detection. 0 objects detected.')
        return [None]


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

    # build the tranlation matrix
    translation_matrix = np.eye(4)
    translation_matrix[3][0:3] = np.array(pose_translation)

    # compute the tranformation matrix
    transformation_matrix = rotation_matrix @ translation_matrix

    # # convert robot's right-handed coordinate system to unreal world's left-handed coordinate system
    # transformation_matrix[:, 1] = -1.0 * transformation_matrix[:, 1]

    return transformation_matrix


def convert_depth_to_pointcloud(observations):
    # convert depth image into point cloud
    depth_image = observations['image_depth']
    image_info = observations['image_depth_info']
    cam_intrinsics = image_info['matrix_intrinsics']
    nx = np.arange(0, image_info['width'])
    ny = np.arange(0, image_info['height'])
    x, y = np.meshgrid(nx, ny)
    fx = cam_intrinsics[0,0]
    fy = cam_intrinsics[1,1]
    cx = cam_intrinsics[0,2]
    cy = cam_intrinsics[1,2]
    x3 = (x - cx) * depth_image * 1 / fx
    y3 = (y - cy) * depth_image * 1 / fy
    # pack the point cloud according to the votenet camera coordinate system
    # which is actually the same as benchbot's
    point3d = np.stack((x3.flatten(), depth_image.flatten(), -y3.flatten()),
                       axis=1)

    return point3d


def votenet_detection(net, observations):
    # build the transformation matrix w_T_c that transforms
    # the point in the camera coordinate system into the world system
    # P_w = P_c * w_T_c, P_c = [x, y, z, 1] a row vector denoting a point location in the camera coordinate
    camera_pose = observations['poses']['camera']
    w_T_c = compute_world_T_camera(camera_pose['rotation_xyzw'],
                                   camera_pose['translation_xyz'])
    # print(np.array([0, 0, 1, 1], dtype=np.float) @ w_T_c)

    point3d = convert_depth_to_pointcloud(observations)

    # forward the point cloud into the votenet trained on SUNRGBD dataset
    detections = votenet_forward(net, point3d)

    # parse the detection results for current frame
    frame_results = []
    if detections[0]:
        detections = detections[0]
        for detection in detections:
            # ignore those classes that are not in ACRV classes
            if detection[0] > net.type2class['desk']:
                continue
            result = {
                "class": net.class2type[detection[0]],
                "class_ID": net.type2class[net.class2type[detection[0]]],
                "confidence": np.float64(detection[2]),
                "box_corners_camera": detection[1]
            }

            # convert the 3D box corners in the camera coordinate system to the world coordinate system
            box_corners_camera = detection[1]
            box_corners_camera = np.concatenate(
                (box_corners_camera,
                 np.ones((box_corners_camera.shape[0], 1),
                         dtype=box_corners_camera.dtype)),
                axis=1)
            box_corners_world = box_corners_camera @ w_T_c
            box_corners_world = box_corners_world[:, 0:3]
            result["box_corners_world"] = box_corners_world
            # compute the 3D box's center roughly
            box_center_world = np.sum(box_corners_world, axis=0) / 8.0
            result["centroid"] = box_center_world
            # compute the 3D box's extent roughly
            box_extent_world = np.sum(
                np.abs(box_corners_world - box_center_world), axis=0) / 4.0
            result["extent"] = box_extent_world

            frame_results.append(result)

    return frame_results


def votenet_nms(all_results, net, class_list):
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

    pick = nms_3d_faster_samecls(boxes_world, nms_iou_cls,
                                 net.eval_config_dict['use_old_type_nms'])
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

    pick = nms_3d_faster(boxes_world, nms_iou,
                         net.eval_config_dict['use_old_type_nms'])
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