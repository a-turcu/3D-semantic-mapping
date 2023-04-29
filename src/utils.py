import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


DEPTH_PATH = "../data_cv2/depth_imgs/image_depth_"
RGB_PATH = "../data_cv2/rgb_imgs/image_rgb_"
INTRINSIC_PATH = "../data_cv2/info_intrinsics.npy"
IMAGE_NUMBER = "19"

def load_image(path):
    img = np.load(path)
    return o3d.geometry.Image(img), img.shape[0], img.shape[1]


def load_intrinsic(path, width, height):
    """" 
    Loads the intrinsic matrices from a .npy file and return them as 
    a list of o3d.camera.PinholeCameraIntrinsic objects
    """
    
    intrinsics = np.load(path)
    intrinsics_list = []

    for i in range(0, intrinsics.shape[0], 9):
        intr = intrinsics[i:i+9]
        fx = intr[0]
        cx = intr[2]
        fy = intr[4]
        cy = intr[5]
        intrinsics_list.append(o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy))
    
    return intrinsics_list


def create_pcd(rgb_img, depth_img, intrinsic):
    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img, depth_img, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd


for i in range(21):
    IMAGE_NUMBER = str(i)
    depth_img, width, height = load_image(DEPTH_PATH + IMAGE_NUMBER + ".npy")
    rgb_img, _, _ = load_image(RGB_PATH + IMAGE_NUMBER + ".npy")
    intrinsic_list = load_intrinsic(INTRINSIC_PATH, width, height)
    
    pcd = create_pcd(rgb_img, depth_img, intrinsic_list[int(IMAGE_NUMBER)])

    o3d.visualization.draw_geometries([pcd])
