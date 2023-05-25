DEPTH_PATH_PASSIVE = "../data/benchbot_data_passive/000000/depth/" 
RGB_PATH_PASSIVE = "../data/benchbot_data_passive/000000/RGB/"
INFO_PATH_PASSIVE = "../data/benchbot_data_passive/000000/RGB_info/"
POSE_PATH_PASSIVE = "../data/benchbot_data_passive/000000/poses/"
CLOUD_PATH = "../data/point_clouds/"

DEPTH_PATH_ACTIVE = "../data/benchbot_data_active/000000/depth/"
RGB_PATH_ACTIVE = "../data/benchbot_data_active/000000/RGB/"
INFO_PATH_ACTIVE = "../data/benchbot_data_active/000000/RGB_info/"
POSE_PATH_ACTIVE = "../data/benchbot_data_active/000000/poses/"

TYPE = "ACTIVE"
# TYPE = "PASSIVE"

# Select the right paths
# ! TESTING ONLY !
if TYPE == "ACTIVE":
    DEPTH_PATH = DEPTH_PATH_ACTIVE
    RGB_PATH = RGB_PATH_ACTIVE
    INFO_PATH = INFO_PATH_ACTIVE
    POSE_PATH = POSE_PATH_ACTIVE
else:
    DEPTH_PATH = DEPTH_PATH_PASSIVE
    RGB_PATH = RGB_PATH_PASSIVE
    INFO_PATH = INFO_PATH_PASSIVE
    POSE_PATH = POSE_PATH_PASSIVE


IMAGE_WIDTH = 720
IMAGE_HEIGHT = 1280
NUM_POINTS = 40000


CHECKPOINT_FILE = "mmdetection3d/checkpoints/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth"
CONFIG_FILE = "mmdetection3d/configs/votenet/votenet_8xb16_sunrgbd-3d.py"
