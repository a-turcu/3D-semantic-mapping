import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
from plyfile import PlyData
import trimesh

DEPTH_PATH = "../data_cv2/depth_imgs/image_depth_"
RGB_PATH = "../data_cv2/rgb_imgs/image_rgb_"
CLOUD_PATH = "../data_cv2/point_clouds/"
INTRINSIC_PATH = "../data_cv2/info_intrinsics.npy"
IMAGE_WIDTH = 720
IMAGE_HEIGHT = 1280


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

#convert_ply("pcd_16.ply", "pcd_16.bin")


#path_demo = ".\mmdetection3d\demo\data\kitti\000008.bin"

pcd = o3d.io.read_point_cloud(CLOUD_PATH + "pcd_16.ply")
#o3d.visualization.draw_geometries([pcd])

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=0.0001)
#o3d.visualization.draw_geometries([voxel_grid])



# for i in range(21):
#     img_nr = str(i)
#     depth_img = load_image(DEPTH_PATH + img_nr + ".npy")
#     rgb_img = load_image(RGB_PATH + img_nr + ".npy")
#     intrinsic_list = load_intrinsic()
    
#     pcd = rgbd_to_pcd(rgb_img, depth_img, intrinsic_list[int(img_nr)])

#     o3d.visualization.draw_geometries([pcd])
    # #save as ply
    # o3d.io.write_point_cloud("pcd_" + img_nr + ".ply", pcd)