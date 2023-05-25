import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pickle
import sys
import time
from global_registration import execute_fast_global_registration, preprocess_point_cloud
IMAGE_WIDTH = 720
IMAGE_HEIGHT = 1280
NUM_POINTS = 40000

# def convert_depth_to_pointcloud(depth_image, intr):
#     # convert depth image into point cloud
#     ny = np.arange(0, IMAGE_WIDTH)
#     nx = np.arange(0, IMAGE_HEIGHT)
#     x, y = np.meshgrid(nx, ny)
#     fx = intr[0][0]
#     cx = intr[0][2]
#     fy = intr[1][1]
#     cy = intr[1][2]
#     x3 = (x - cx) * depth_image * 1 / fx
#     y3 = (y - cy) * depth_image * 1 / fy
#     # pack the point cloud according to the votenet camera coordinate system
#     # which is actually the same as benchbot's
#     point3d = np.stack((x3.flatten(), depth_image.flatten(), -y3.flatten()),
#                        axis=1)
#
#     return point3d


# def create_pcd(rgb_img, depth_img, intrinsic):
#     rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img, depth_img, convert_rgb_to_intensity=False)
#     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic)
#     pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
#
#     return pcd


def prepare_pcd(voxel_size, pcd, downsample=True):
    # print(":: Downsample with a voxel size %.6f." % voxel_size)
    pcd_down = pcd
    if downsample:
        pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.6f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.6f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )

    return pcd_down, pcd_fpfh


# def get_intrinsics(observations):
#     intr = observations["image_depth_info"]
#     width = intr["width"]
#     height = intr["height"]
#     intr = intr["matrix_intrinsics"]
#     fx = intr[0][0]
#     cx = intr[0][2]
#     fy = intr[1][1]
#     cy = intr[1][2]
#     return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def pcd_pipeline(observations):
    # rgb = o3d.geometry.Image(observations["image_rgb"])
    
    # depth = observations["image_depth"]
    # depth[depth >= 5] = 0
    # depth = o3d.geometry.Image(depth)

    # intrinsics = get_intrinsics(observations)
    # pcd = create_pcd(rgb, depth, intrinsics)

    # return pcd
    depth_img, rgb_np = observations["image_depth"], observations["image_rgb"]   
    # threshold depth image as to remove unwanted background
    depth_img[depth_img >= 5] = 0
    intrinsics = observations["image_depth_info"]["matrix_intrinsics"]

    pcd = convert_depth_to_pointcloud(depth_img, intrinsics)

    # convert numpy pcd to open3d pcd
    points = pcd[:, :3] / 1000
    colors = rgb_np.reshape(-1, 3) / 255.0
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(points)
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors)

    return pcd_o3d


def refine_registration(source, target, result, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result


def combine_PCD(combined_pcd, new_obs):
    voxel_size = 0.00008

    new_pcd = pcd_pipeline(new_obs)

    new_pcd_down, new_pcd_fpfh = prepare_pcd(voxel_size, new_pcd)
    combined_pcd_down, combined_pcd_fpfh = prepare_pcd(voxel_size, combined_pcd)

    results_registration = execute_fast_global_registration(
        combined_pcd_down, new_pcd_down, combined_pcd_fpfh, new_pcd_fpfh, voxel_size
    )

    results_registration = refine_registration(combined_pcd_down, new_pcd_down, results_registration, voxel_size)
    combined_pcd = combined_pcd.transform(results_registration.transformation) + new_pcd

    #o3d.visualization.draw_geometries([combined_pcd])

    return combined_pcd


def save_pcd(pcd):
    o3d.io.write_point_cloud("pointcloud_map.ply", pcd)
    # numpy_pcd = np.asarray(combined_pcd.points)
    # np.save("pointcloud_map_numpy.npy", numpy_pcd)



