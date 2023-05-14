import os
import numpy as np
import open3d as o3d
import pandas as pd
from plyfile import PlyData
import trimesh
import pickle
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.registry import VISUALIZERS

from const import *


#! Unused but could be useful in the future
def rgbd_to_pcd(rgb_img: o3d.geometry.Image, depth_img: o3d.geometry.Image, intrinsic: o3d.camera.PinholeCameraIntrinsic):
    """
    Converts a RGBD image to a point cloud using o3d
    """
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img, depth_img, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd


# TODO try saving to bin instead of npy
def create_pcd(depth_img, rgb_np, intrinsics, filename, save=True):
    """
    Converts a depth image to a point cloud and adds color afterwards
    Optionally save it as numpy array
    ! No o3d used
    """

    pcd = convert_depth_to_pointcloud(depth_img, intrinsics)
    #add color
    pcd = np.concatenate((pcd, rgb_np.reshape(-1, 3)), axis=1)
    pcd = pcd.astype(np.float32)

    if save:
        # if path does not exist, create it
        if not os.path.exists(CLOUD_PATH):
            os.makedirs(CLOUD_PATH)
        np.save(CLOUD_PATH + filename + ".npy", pcd)
    
    return pcd


# load pkl file
def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


#! Unused
def convert_ply(input_path, output_path):
    plydata = PlyData.read(input_path)  # read file
    data = plydata.elements[0].data  # read data
    data_pd = pd.DataFrame(data)  # convert to DataFrame
    data_np = np.zeros(data_pd.shape, dtype=np.float)  # initialize array to store data
    property_names = data[0].dtype.names  # read names of properties
    for i, name in enumerate(property_names):  # read data by property
        data_np[:, i] = data_pd[name]
    data_np.astype(np.float32).tofile(output_path)


#! Unused
def to_ply(input_path, output_path, original_type):
    mesh = trimesh.load(input_path, file_type=original_type)  # read file
    mesh.export(output_path, file_type='ply')  # convert to ply


#! Unused
def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


#! Unused
def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:, 0:3]  # do not use color for now
    floor_height = np.percentile(point_cloud[:, 2], 0.99)
    height = point_cloud[:, 2] - floor_height
    
    point_cloud = np.concatenate(
        [point_cloud, np.expand_dims(height, 1)], 1)  # (N,4) or (N,7)
    point_cloud = random_sampling(point_cloud, NUM_POINTS)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0)  # (1,n,4)
    return pc


def convert_depth_to_pointcloud(depth_image, intr):
    # convert depth image into point cloud
    ny = np.arange(0, IMAGE_WIDTH)
    nx = np.arange(0, IMAGE_HEIGHT)
    x, y = np.meshgrid(nx, ny)
    fx = intr[0]
    cx = intr[2]
    fy = intr[4]
    cy = intr[5]
    x3 = (x - cx) * depth_image * 1 / fx
    y3 = (y - cy) * depth_image * 1 / fy
    # pack the point cloud according to the votenet camera coordinate system
    # which is actually the same as benchbot's
    point3d = np.stack((x3.flatten(), depth_image.flatten(), -y3.flatten()),
                       axis=1)

    return point3d


def vis_o3d(pcd):
    pdc_o3d = o3d.geometry.PointCloud()
    pdc_o3d.points = o3d.utility.Vector3dVector(pcd)
    o3d.visualization.draw_geometries([pdc_o3d])


def vis_mm(model, data, result, threshold=0):

    points = data['inputs']['points']
    data_input = dict(points=points)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    visualizer.add_datasample(
        'result',
        data_input,
        data_sample=result,
        draw_gt=False,
        show=True,
        wait_time=0,
        out_file='test.png',
        pred_score_thr=threshold,
        vis_task='lidar_det')

