import numpy as np

from mmdet3d.apis import inference_detector, init_model
import os

from const import *

from mmdet3d.visualization import Det3DLocalVisualizer
from utils import create_pcd, load_pkl, vis_mm, random_sampling
import open3d as o3d
import json






def main():

    # filename = "global_pcd"
    #
    # pcd_path = CLOUD_PATH + filename + ".npy"
    # pcd = np.load(pcd_path)
    #
    # pcd = transform_pcd(pcd)

    transformed_path = CLOUD_PATH + "global_pcd_transformed.npy"
    # np.save(transformed_path, pcd)

    threshold = 0.7
    model = init_model(CONFIG_FILE, CHCKPOINT_FILE)
    result, data = inference_detector(model, transformed_path)
    results = analyze_result(result, threshold)
    print(results)
    save_filename = "results.json"
    with open(save_filename, 'w') as f:
        json.dump(results, f)

    # # bed
    # box_bed = DepthInstance3DBoxes(torch.tensor([[2.20074341, 0.05944466, 0.44632973, 2.29851074, 1.85151764, 0.89577698, 0]]))
    #
    # # sink
    # box_sink = DepthInstance3DBoxes(torch.tensor([[-0.58994232, 3.07765717, 0.82831787, 0.32866461999999996, 0.27531434, 0.14530166, 0]]))
    #
    # # table 1
    # box_table1 = DepthInstance3DBoxes(torch.tensor([[0.47003494, -0.53999237, 0.17274715000000002, 0.60470368, 0.40477172000000006, 0.34549430000000003, 0]]))
    #
    # # table 2
    # box_table2 = DepthInstance3DBoxes(torch.tensor([[3.11474518, 1.08274681, 0.17274715000000002, 0.40477112, 0.60470322, 0.34549430000000003, 0]]))
    #
    # # center
    # box_center = DepthInstance3DBoxes(torch.tensor([[-0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0]]))

    # result.pred_instances_3d.bboxes_3d.tensor[0, :] = box_bed.tensor
    # result.pred_instances_3d.bboxes_3d.tensor[1, :] = box_sink.tensor
    # result.pred_instances_3d.bboxes_3d.tensor[2, :] = box_table1.tensor
    # result.pred_instances_3d.bboxes_3d.tensor[3, :] = box_table2.tensor
    # result.pred_instances_3d.bboxes_3d.tensor[4:, :] = box_center.tensor
    #
    # result.pred_instances_3d.scores_3d[0] = 0.99
    # result.pred_instances_3d.scores_3d[1] = 0.99
    # result.pred_instances_3d.scores_3d[2] = 0.99
    # result.pred_instances_3d.scores_3d[3] = 0.99


    #vis_mm(model, data, result, threshold)

    print("ok")


if __name__ == "__main__":
    main()