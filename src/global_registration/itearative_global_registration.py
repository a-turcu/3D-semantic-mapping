import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pickle
import sys
import time

from PIL import Image

from global_registration import execute_fast_global_registration, preprocess_point_cloud
from mmdet3d.registry import VISUALIZERS
from mmdet3d.apis import inference_detector, init_model
from solution.utils_pcd import load_pkl, convert_depth_to_pointcloud, random_sampling
from solution.main import load_data

DEPTH_PATH = "../data/data_global/depth/0000"
RGB_PATH = "../data/data_global/RGB/0000"
INTRINSIC_PATH = "../data/data_global/RGB_info/0000"


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
    #pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

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


def load_data(filename):
    if filename < 10:
        filename = "0" + str(filename)
    depth_img = np.load(DEPTH_PATH + str(filename) + ".npy")
    rgb_img = Image.open(RGB_PATH + str(filename) + ".png")
    rgb_np = np.asarray(rgb_img, dtype=np.float32)
    info = load_pkl(INTRINSIC_PATH + str(filename) + ".pkl")
    intrinsics = info['matrix_intrinsics']
    intrinsics = intrinsics.reshape(9)
    intrinsics = intrinsics.astype(np.float32)
   # poses = load_pkl(POSE_PATH + filename + ".pkl")

    return depth_img, rgb_np, intrinsics, None


def pcd_pipeline(image_number):
    # rgb = load_png(image_number)
    # depth = load_depth(image_number)
    # intrinsics = load_intrinsics(image_number)
    # pcd_original = create_pcd(rgb, depth, intrinsics)

    depth_img, rgb_np, intrinsics, poses = load_data(image_number)
    depth_img[depth_img >= 5] = 0
    pcd = convert_depth_to_pointcloud(depth_img, intrinsics)

    #pcd = random_sampling(pcd, 30000)

    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd[:, :3] / 1000)
    pcd_o3d.colors = o3d.utility.Vector3dVector(rgb_np.reshape(-1, 3) / 255.0)

    # o3d.visualization.draw_geometries([pcd_original])
    # o3d.visualization.draw_geometries([pcd_o3d])

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

    challenge_classes = classes_sunrgbd[:6]
    challenge_classes[5] = 'table'

    for i in labels_3d:
        print(classes_sunrgbd[i])


def main():
    voxel_size = 0.00008
    voxel_size_combined = 0.0001
    combined_pcd = pcd_pipeline(image_number=0)

    CHCKPOINT_FILE = "mmdetection3d/checkpoints/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth"
    CONFIG_FILE = "mmdetection3d/configs/votenet/votenet_8xb16_sunrgbd-3d.py"
    model = init_model(CONFIG_FILE, CHCKPOINT_FILE)

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
        #print(combined_pcd)
        pcd = np.asarray(combined_pcd.points)
        rgb_np = np.asarray(combined_pcd.colors)
        pcd = np.concatenate((pcd, rgb_np.reshape(-1, 3)), axis=1)
        pcd = pcd.astype(np.float32)
        pcd[:, :3] = pcd[:, :3] * 1000

        pcd_path = "pcd.npy"
        np.save(pcd_path, pcd)

        # if img_num % 20 == 0:
        #     result, data = inference_detector(model, pcd_path)
        #
        #     points = data['inputs']['points']
        #     data_input = dict(points=points)
        #     visualizer = VISUALIZERS.build(model.cfg.visualizer)
        #     visualizer.dataset_meta = model.dataset_meta
        #
        #     analyze_result(result, threshold=0.8)
        #
        #     visualizer.add_datasample(
        #         'result',
        #         data_input,
        #         data_sample=result,
        #         draw_gt=False,
        #         show=True,
        #         wait_time=0,
        #         out_file='test.png',
        #         pred_score_thr=0.8,
        #         vis_task='lidar_det')

            # o3d.visualization.draw_geometries([combined_pcd])

    result, data = inference_detector(model, pcd_path)

    points = data['inputs']['points']
    data_input = dict(points=points)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    analyze_result(result, threshold=0.8)

    visualizer.add_datasample(
        'result',
        data_input,
        data_sample=result,
        draw_gt=False,
        show=True,
        wait_time=0,
        out_file='test.png',
        pred_score_thr=0.8,
        vis_task='lidar_det')


#o3d.visualization.draw_geometries([combined_pcd])

    #o3d.io.write_point_cloud("pointcloud_map.ply", combined_pcd)
    
    # pcd = np.asarray(combined_pcd.points)
    # rgb_np = np.asarray(combined_pcd.colors)
    # pcd = np.concatenate((pcd, rgb_np.reshape(-1, 3)), axis=1)
    # pcd = pcd.astype(np.float32)
    # print(pcd.shape)
    # np.save("pointcloud_map_rgb.npy", pcd)


if __name__ == "__main__":
    main()
