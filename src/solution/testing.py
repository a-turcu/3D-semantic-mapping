"""
!!! FILE CREATED FOR TESTING STUFF OUT !!!
"""

from scipy.spatial.transform import Rotation
from mmdet3d.registry import VISUALIZERS
from mmdet3d.apis import inference_detector, init_model
from solution.utils_pcd import load_pkl, convert_depth_to_pointcloud, random_sampling
from solution.main import load_data

from main import *


def vis_mm_2(pcd, boxes=None):
    visualizer = Det3DLocalVisualizer(pcd_mode=2)
    visualizer.set_points(pcd, mode='xyzrgb')
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

    # build the tranlation matrix
    translation_matrix = np.eye(4)
    translation_matrix[3][0:3] = np.array(pose_translation)

    # compute the tranformation matrix
    transformation_matrix = rotation_matrix @ translation_matrix

    # # convert robot's right-handed coordinate system to unreal world's left-handed coordinate system
    # transformation_matrix[:, 1] = -1.0 * transformation_matrix[:, 1]

    return transformation_matrix


# def camera_to_world_coord(boxes):
#     # convert the 3D box corners in the camera coordinate system to the world coordinate system
#     box_corners_camera = detection[1]
#     box_corners_camera = np.concatenate(
#         (box_corners_camera,
#          np.ones((box_corners_camera.shape[0], 1),
#                  dtype=box_corners_camera.dtype)),
#         axis=1)
#     box_corners_world = box_corners_camera @ w_T_c
#     box_corners_world = box_corners_world[:, 0:3]
#     result["box_corners_world"] = box_corners_world
#     # compute the 3D box's center roughly
#     box_center_world = np.sum(box_corners_world, axis=0) / 8.0
#     result["centroid"] = box_center_world
#     # compute the 3D box's extent roughly
#     box_extent_world = np.sum(
#         np.abs(box_corners_world - box_center_world), axis=0) / 4.0
#     result["extent"] = box_extent_world


def main():

    # filename = "000016"
    # pcd_path = CLOUD_PATH + filename + ".npy"
    #
    # _, _, _, poses = load_data(filename)
    #
    # # camera_pose = poses['camera']
    # # w_T_c = compute_world_T_camera(camera_pose['rotation_xyzw'],
    # #                              camera_pose['translation_xyz'])
    #
    #
    # # model = init_model(CONFIG_FILE, CHCKPOINT_FILE)
    # # result, data = inference_detector(model, pcd_path)
    #
    # # analyze_result(result, threshold=0.5)
    # # vis_mm(model, data, result, threshold=0.5)
    #
    #
    # for i in range(len(listdir(CLOUD_PATH))):
    #     if i < 10:
    #         filename = "00000" + str(i)
    #     else:
    #         filename = "0000" + str(i)
    #
    #     pcd_path = CLOUD_PATH + filename + ".npy"
    #     model = init_model(CONFIG_FILE, CHCKPOINT_FILE)
    #     result, data = inference_detector(model, pcd_path)
    #     vis_mm(model, data, result, threshold=0.5)
    #
    # #depth_img, rgb_np, intrinsics, poses = load_data(filename)
    # #create_pcd(depth_img, rgb_np, intrinsics, filename)
    #
    # print("ok")

    pcd_path = "pcd.npy"
    CHCKPOINT_FILE = "mmdetection3d/checkpoints/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth"
    CONFIG_FILE = "mmdetection3d/configs/votenet/votenet_8xb16_sunrgbd-3d.py"
    model = init_model(CONFIG_FILE, CHCKPOINT_FILE)

    result, data = inference_detector(model, pcd_path)

    points = data['inputs']['points']
    data_input = dict(points=points)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    analyze_result(result, threshold=0.5)

    visualizer.add_datasample(
        'result',
        data_input,
        data_sample=result,
        draw_gt=False,
        show=True,
        wait_time=0,
        out_file='test.png',
        pred_score_thr=0.5,
        vis_task='lidar_det')

if __name__ == "__main__":
    main()
