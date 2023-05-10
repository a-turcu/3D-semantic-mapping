import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
from plyfile import PlyData
import trimesh
import pickle
import scipy.io as sio
from PIL import Image


DEPTH_PATH = "../data/benchbot_data_passive/miniroom_1/000000/depth/" 
DEPTH_MAT_PATH = "../data/benchbot_data_passive/miniroom_1/000000/depth_mat/"
RGB_PATH = "../data/benchbot_data_passive/miniroom_1/000000/RGB/"
CLOUD_PATH = "../data/benchbot_data_passive/miniroom_1/000000/point_clouds/"
INFO_PATH = "../data/benchbot_data_passive/miniroom_1/000000/RGB_info/"


IMAGE_WIDTH = 720
IMAGE_HEIGHT = 1280
NUM_POINTS = 40000


#! UNUSED O3D FUNCTIONS

# def load_image(path):
#     img = np.load(path)
#     return o3d.geometry.Image(img)


# def load_intrinsic():
#     """" 
#     Loads the intrinsic matrices from a .npy file and return them as 
#     a list of o3d.camera.PinholeCameraIntrinsic objects
#     """
#     intrinsics = np.load(INTRINSIC_PATH)
#     intrinsics_list = []

#     for i in range(0, intrinsics.shape[0], 9):
#         intr = intrinsics[i:i+9]
#         fx = intr[0]
#         cx = intr[2]
#         fy = intr[4]
#         cy = intr[5]
        
#         intrinsics_list.append(o3d.camera.PinholeCameraIntrinsic(IMAGE_WIDTH, IMAGE_HEIGHT, fx, fy, cx, cy))
#     return intrinsics_list


# def rgbd_to_pcd(rgb_img, depth_img, intrinsic):
#     """
#     Converts a RGBD image to a point cloud
#     """
#     rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img, depth_img, convert_rgb_to_intensity=False)
#     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic)
#     pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
#     return pcd


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


# load pkl file
def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def visualize_pcd(pcd):
    pdc_o3d = o3d.geometry.PointCloud()
    pdc_o3d.points = o3d.utility.Vector3dVector(pcd)
    o3d.visualization.draw_geometries([pdc_o3d])

def visualize_color_pcd(pcd, rgb):
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_o3d.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3))
    o3d.visualization.draw_geometries([pcd_o3d])



def save_mat(pcd, filename):
    sio.savemat(DEPTH_MAT_PATH + filename + ".mat", {"instance": pcd})


def main():

    # for i in range(10, 48):
    #     filename = "0000" + str(i)
    #     depth_img = np.load(DEPTH_PATH + filename + ".npy")
    #     save_mat(depth_img, filename)

    filename = "000016"
    depth_img = np.load(DEPTH_PATH + filename + ".npy")
    save_mat(depth_img, filename)

    rgb_img = Image.open(RGB_PATH + filename + ".png")
    rgb_np = np.asarray(rgb_img, dtype=np.float32)

    info = load_pkl(INFO_PATH + filename + ".pkl")
    intrinsics = info['matrix_intrinsics']
    intrinsics = intrinsics.reshape(9)
    pcd = convert_depth_to_pointcloud(depth_img, intrinsics)

    

    # convert pcd to float
    pcd_np = pcd.astype(np.float32)
    # visualize_pcd(pcd_np)
    visualize_color_pcd(pcd_np, rgb_np)
    # sampled_pcd = random_sampling(pcd, NUM_POINTS)
    # print(sampled_pcd.shape)
    # visualize_pcd(sampled_pcd)




if __name__ == "__main__":
    main()

