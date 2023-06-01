import os
import numpy as np
import open3d as o3d
import pickle
#from mmdet3d.registry import VISUALIZERS
from PIL import Image

from const import *


def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


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
    point3d = np.stack((x3.flatten(), depth_image.flatten(), -y3.flatten()), axis=1)
    return point3d


def vis_o3d(pcd):
    pdc_o3d = o3d.geometry.PointCloud()
    pdc_o3d.points = o3d.utility.Vector3dVector(pcd)
    o3d.visualization.draw_geometries([pdc_o3d])


def numpy_to_open3d(pcd, rgb_np):
    # convert numpy pcd to open3d pcd
    points = pcd[:, :3] / 1000
    colors = rgb_np.reshape(-1, 3) / 255.0
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(points)
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
    return pcd_o3d


def open3d_to_numpy(pcd_o3d):
    # convert open3d to numpy
    pcd = np.asarray(pcd_o3d.points)
    rgb_np = np.asarray(pcd_o3d.colors)
    pcd = np.concatenate((pcd, rgb_np.reshape(-1, 3)), axis=1)
    pcd = pcd.astype(np.float32)
    pcd[:, :3] = pcd[:, :3] * 1000
    return pcd


def transform_pcd(pcd):

    # sample pcd for faster testing
    pcd = random_sampling(pcd, 500_000)

    # numpy to o3d
    pcd_o3d = numpy_to_open3d(pcd, pcd[:, 3:])

    # transform
    R = pcd_o3d.get_rotation_matrix_from_xyz((0, 0, 0.63*np.pi))
    pcd_o3d = pcd_o3d.rotate(R, center=(0, 0, 0))
    pcd_o3d = pcd_o3d.translate((0.0011, 0.0012, 0.0012))

    # o3d to numpy
    pcd = open3d_to_numpy(pcd_o3d)

    return pcd