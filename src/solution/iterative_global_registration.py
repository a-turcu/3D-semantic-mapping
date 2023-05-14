import numpy as np
import open3d as o3d
import time

from main import load_data
from const import *
from utils_pcd import convert_depth_to_pointcloud

# fairly small, might be too detailed
VOXEL_SIZE = 0.00008
VOXEL_SIZE_COMBINED = 0.0001


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


def pcd_pipeline(filename):
    """
    Load and create numpy pointcloud and then convert it to open3d pointcloud
    Conversion is necessary for the registration pipeline
    """	

    depth_img, rgb_np, intrinsics, poses = load_data(filename)    
    # threshold depth image as to remove unwanted background
    depth_img[depth_img >= 5] = 0

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
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result


def execute_fast_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 0.5
    print(
        ":: Apply fast global registration with distance threshold %.3f"
        % distance_threshold
    )
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold
        ),
    )
    return result


def create_global_pointcloud(save_path):
    
    combined_pcd = pcd_pipeline("000000")

    start = time.time()
    for img_num in range(1, 56):
        print("Processing image: " + str(img_num) + ", Time so far: " + str(time.time() - start))
        filename = "0" * (6 - len(str(img_num))) + str(img_num)
        
        new_pcd = pcd_pipeline(filename)

        new_pcd_down, new_pcd_fpfh = prepare_pcd(VOXEL_SIZE, new_pcd)
        combined_pcd_down, combined_pcd_fpfh = prepare_pcd(VOXEL_SIZE_COMBINED, combined_pcd)

        results_registration = execute_fast_global_registration(
            combined_pcd_down, new_pcd_down, combined_pcd_fpfh, new_pcd_fpfh, VOXEL_SIZE
        )

        results_registration = refine_registration(combined_pcd_down, new_pcd_down, results_registration, VOXEL_SIZE)
        combined_pcd = combined_pcd.transform(results_registration.transformation) + new_pcd
    
    # convert back to numpy and save
    pcd = np.asarray(combined_pcd.points)
    rgb_np = np.asarray(combined_pcd.colors)
    pcd = np.concatenate((pcd, rgb_np.reshape(-1, 3)), axis=1)
    pcd = pcd.astype(np.float32)
    pcd[:, :3] = pcd[:, :3] * 1000

    np.save(save_path, pcd)
