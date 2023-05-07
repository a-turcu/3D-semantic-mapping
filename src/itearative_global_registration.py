import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pickle
import sys
import time
from global_registration import execute_fast_global_registration, preprocess_point_cloud

DEPTH_PATH = "D:/Master/Block5/CV2/Data_Smooth/depth/0000"
RGB_PATH = "D:/Master/Block5/CV2/Data_Smooth/RGB/0000"
INTRINSIC_PATH = "D:/Master/Block5/CV2/Data_Smooth/RGB_info/0000"


def load_png(image_number):
    if image_number < 10:
        return o3d.io.read_image(RGB_PATH + "0" + str(image_number) + ".png")
    else:
        return o3d.io.read_image(RGB_PATH + str(image_number) + ".png")


def load_depth(image_number):
    if image_number < 10:
        depth = np.load(DEPTH_PATH + "0" + str(image_number) + ".npy")
    else:
        depth = np.load(DEPTH_PATH + str(image_number) + ".npy")

    # try some depth treshold in meters
    # np.clip(depth, 0, 2, out=depth)
    # replace all max values with 0
    depth[depth >= 3] = 0

    return o3d.geometry.Image(depth)


def load_intrinsics(image_number):
    if image_number < 10:
        path = INTRINSIC_PATH + "0" + str(image_number) + ".pkl"
    else:
        path = INTRINSIC_PATH + str(image_number) + ".pkl"

    with open(path, "rb") as f:
        intr = pickle.load(f)
        width = intr["width"]
        height = intr["height"]
        intr = intr["matrix_intrinsics"]
        fx = intr[0][0]
        cx = intr[0][2]
        fy = intr[1][1]
        cy = intr[1][2]
        return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def create_pcd(rgb_img, depth_img, intrinsic):
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img, depth_img, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd


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


def pcd_pipeline(image_number):
    rgb = load_png(image_number)
    depth = load_depth(image_number)
    intrinsics = load_intrinsics(image_number)
    pcd = create_pcd(rgb, depth, intrinsics)

    return pcd


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


def main():
    voxel_size = 0.00008
    voxel_size_combined = 0.0001
    combined_pcd = pcd_pipeline(image_number=0)

    start = time.time()
    for img_num in range(1, 56):
        print("Processing image: " + str(img_num) + ", Time so far: " + str(time.time() - start))
        new_pcd = pcd_pipeline(image_number=img_num)

        new_pcd_down, new_pcd_fpfh = prepare_pcd(voxel_size, new_pcd)
        combined_pcd_down, combined_pcd_fpfh = prepare_pcd(voxel_size_combined, combined_pcd)

        results_registration = execute_fast_global_registration(
            combined_pcd_down, new_pcd_down, combined_pcd_fpfh, new_pcd_fpfh, voxel_size
        )

        results_registration = refine_registration(combined_pcd_down, new_pcd_down, results_registration, voxel_size)
        combined_pcd = combined_pcd.transform(results_registration.transformation) + new_pcd
        print(combined_pcd)

    o3d.visualization.draw_geometries([combined_pcd])

    o3d.io.write_point_cloud("pointcloud_map.ply", combined_pcd)
    
    numpy_pcd = np.asarray(combined_pcd.points)
    
    np.save("pointcloud_map_numpy.npy", numpy_pcd)


if __name__ == "__main__":
    main()
