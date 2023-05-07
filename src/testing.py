from mmdet3d.apis import inference_detector, init_model
from mmdet3d.registry import VISUALIZERS
from mmdet3d.visualization import Det3DLocalVisualizer

#path = 'demo/data/kitti/000008.bin'
path = "../../data_cv2/point_clouds/pcd_16.npy"
config_file = 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'
checkpoint_file = 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth'
device = 'cpu'
model = init_model(config_file, checkpoint_file, device=device)
result, data = inference_detector(model, path)

points = data['inputs']['points']
data_input = dict(points=points)

print(result)

# convert points to numpy array
# points = points.numpy()
# visualizer = Det3DLocalVisualizer()
# visualizer.set_points(points)
# visualizer.show()

#visualizer = VISUALIZERS.build(model.cfg.visualizer)
#visualizer.dataset_meta = model.dataset_meta
# visualizer.add_datasample(
#     'result',
#     data_input,
#     data_sample=result,
#     draw_gt=False,
#     show=False,
#     wait_time=0,
#     out_file='test.png',
#     pred_score_thr=0.0,
#     vis_task='lidar_det')