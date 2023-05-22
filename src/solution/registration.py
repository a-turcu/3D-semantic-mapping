import open3d as o3d
import time

from utils_pcd import load_data, convert_depth_to_pointcloud, numpy_to_open3d


VOXEL_SIZE = 0.00008



def load_pcd_pipeline(filename, observations=None, mode="offline"):
    """
    Load and create numpy pointcloud and then convert it to open3d pointcloud
    Conversion is necessary for the registration pipeline
    """
    if mode == "offline":
        depth_img, rgb_np, intrinsics, poses = load_data(filename)
    else:
        depth_img, rgb_np = observations["image_depth"], observations["image_rgb"]
        intrinsics = observations["image_depth_info"]["matrix_intrinsics"].reshape(9)

    # threshold depth image as to remove unwanted background
    depth_img[depth_img >= 5] = 0

    pcd = convert_depth_to_pointcloud(depth_img, intrinsics)
    pcd_o3d = numpy_to_open3d(pcd, rgb_np)

    return pcd_o3d


def prepare_pcd(voxel_size, pcd):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )

    return pcd_down, pcd_fpfh


def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
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


def refine_registration(source, target, result, voxel_size):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result


def combine_PCD(combined_pcd, new_pcd, voxel_size=0.00008):

    new_pcd_down, new_pcd_fpfh = prepare_pcd(voxel_size, new_pcd)
    combined_pcd_down, combined_pcd_fpfh = prepare_pcd(voxel_size, combined_pcd)

    results_registration = execute_fast_global_registration(
        combined_pcd_down, new_pcd_down, combined_pcd_fpfh, new_pcd_fpfh, voxel_size
    )

    results_registration = refine_registration(combined_pcd_down, new_pcd_down, results_registration, voxel_size)
    combined_pcd = combined_pcd.transform(results_registration.transformation) + new_pcd

    return combined_pcd


def iterative_global_registration(save_path):

    combined_pcd= load_pcd_pipeline("000000")
    o3d.visualization.draw_geometries([combined_pcd])

    start = time.time()
    for img_num in range(1, 56):
        print("Processing image: " + str(img_num) + ", Time so far: " + str(time.time() - start))
        filename = "0" * (6 - len(str(img_num))) + str(img_num)
        new_pcd = load_pcd_pipeline(filename)
        combined_pcd = combine_PCD(combined_pcd, new_pcd, VOXEL_SIZE)

    o3d.visualization.draw_geometries([combined_pcd])
    o3d.io.write_point_cloud(save_path + ".ply", combined_pcd)


iterative_global_registration("data/point_clouds/iterative_global_pcd")