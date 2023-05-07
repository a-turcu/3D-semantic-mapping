import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
from plyfile import PlyData
import trimesh
import pickle

DEPTH_PATH = "../data_cv2/raw_data/depth_imgs/image_depth_"
RGB_PATH = "../data_cv2/raw_data/rgb_imgs/image_rgb_"
CLOUD_PATH = "../data_cv2/point_clouds/"
INTRINSIC_PATH = "../data_cv2/raw_data/info_intrinsics.npy"
IMAGE_WIDTH = 720
IMAGE_HEIGHT = 1280
NUM_POINTS = 40000

def load_image(path):
    img = np.load(path)
    return o3d.geometry.Image(img)


def load_intrinsic():
    """" 
    Loads the intrinsic matrices from a .npy file and return them as 
    a list of o3d.camera.PinholeCameraIntrinsic objects
    """
    intrinsics = np.load(INTRINSIC_PATH)
    intrinsics_list = []

    for i in range(0, intrinsics.shape[0], 9):
        intr = intrinsics[i:i+9]
        fx = intr[0]
        cx = intr[2]
        fy = intr[4]
        cy = intr[5]
        intrinsics_list.append(o3d.camera.PinholeCameraIntrinsic(IMAGE_WIDTH, IMAGE_HEIGHT, fx, fy, cx, cy))
    return intrinsics_list


def rgbd_to_pcd(rgb_img, depth_img, intrinsic):
    """
    Converts a RGBD image to a point cloud
    """
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img, depth_img, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd


def convert_ply(input_path, output_path):
    plydata = PlyData.read(input_path)  # read file
    data = plydata.elements[0].data  # read data
    data_pd = pd.DataFrame(data)  # convert to DataFrame
    data_np = np.zeros(data_pd.shape, dtype=np.float)  # initialize array to store data
    property_names = data[0].dtype.names  # read names of properties
    for i, name in enumerate(
            property_names):  # read data by property
        data_np[:, i] = data_pd[name]
    data_np.astype(np.float32).tofile(output_path)


def to_ply(input_path, output_path, original_type):
    mesh = trimesh.load(input_path, file_type=original_type)  # read file
    mesh.export(output_path, file_type='ply')  # convert to ply


def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


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


# pcd = o3d.io.read_point_cloud(CLOUD_PATH + "pcd_16.ply")
# o3d.visualization.draw_geometries([pcd])


#depth_img = load_image(DEPTH_PATH + "16.npy")
# depth_img = np.load(DEPTH_PATH + "16.npy")
# intrinsic_list = load_intrinsic()

# pcd_np = convert_depth_to_pointcloud(depth_img, intrinsic_list[16])
# prepped_pcd = preprocess_point_cloud(pcd_np)
# np.save(CLOUD_PATH + "pcd_16.npy", prepped_pcd)

#pcd_o3d = o3d.geometry.PointCloud()
#pcd_o3d.points = o3d.utility.Vector3dVector(prepped_pcd)
#o3d.visualization.draw_geometries([pcd_o3d])