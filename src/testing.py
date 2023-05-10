from mmdet3d.apis import inference_detector, init_model, inference_segmentor
from mmdet3d.registry import VISUALIZERS
from mmdet3d.visualization import Det3DLocalVisualizer
import open3d as o3d
import numpy as np

path = 'mmdetection3d/data/sunrgbd/points/005051.bin'

#path = "../data_cv2/point_clouds/pcd_16.npy"

# config_file = 'mmdetection3d/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
# checkpoint_file = 'mmdetection3d/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'

# checkpoint_file = "mmdetection3d/checkpoints/h3dnet_3x8_scannet-3d-18class_20210824_003149-414bd304.pth"
# config_file = "mmdetection3d/configs/h3dnet/h3dnet_8xb3_scannet-seg.py"

# checkpoint_file = "mmdetection3d/checkpoints/votenet_8x8_scannet-3d-18class_20210823_234503-cf8134fa.pth" 
# config_file = "mmdetection3d/configs/votenet/votenet_8xb8_scannet-3d.py"

checkpoint_file = "mmdetection3d/checkpoints/votenet_16x8_sunrgbd-3d-10class_20210820_162823-bf11f014.pth"
config_file = "mmdetection3d/configs/votenet/votenet_8xb16_sunrgbd-3d.py"

# checkpoint_file = "mmdetection3d/checkpoints/fcaf3d_8x2_sunrgbd-3d-10class_20220805_165017.pth"
# config_file = "mmdetection3d/configs/fcaf3d/fcaf3d_2xb8_sunrgbd-3d-10class.py"

model = init_model(config_file, checkpoint_file)
result, data = inference_segmentor(model, path)

points = data['inputs']['points']

data_input = dict(points=points)

boxes = result.pred_instances_3d.bboxes_3d
#print(boxes.tensor.shape)

# points = points.numpy()

# points = np.load(path2)
# visualizer = Det3DLocalVisualizer()
# visualizer.set_points(points)
# visualizer.draw_bboxes_3d(boxes)
# visualizer.show()




visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta
visualizer.add_datasample(
    'result',
    data_input,
    data_sample=result,
    draw_gt=False,
    show=True,
    wait_time=0,
    out_file='test.png',
    pred_score_thr=0.0,
    vis_task='lidar_det')