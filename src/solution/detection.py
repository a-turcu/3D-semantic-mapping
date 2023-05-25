from mmdet3d.registry import VISUALIZERS
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.apis import inference_detector

DEFAULT_CLASS_LIST = [
    'bottle', 'cup', 'knife', 'bowl', 'wine glass', 'fork', 'spoon', 'banana',
    'apple', 'orange', 'cake', 'potted plant', 'mouse', 'keyboard', 'laptop',
    'cell phone', 'book', 'clock', 'chair', 'table', 'couch', 'bed', 'toilet',
    'tv', 'microwave', 'toaster', 'refrigerator', 'oven', 'sink', 'person',
    'background'
]


def create_empty(class_list=None):
    return {
        'objects': [],
        'class_list': DEFAULT_CLASS_LIST if class_list is None else class_list
    }


def create_empty_object(num_classes=None):
    return {
        'label_probs': [0] *
                       (len(DEFAULT_CLASS_LIST) if num_classes is None else num_classes),
        'centroid': [0] * 3,
        'extent': [0] * 3
    }


def jsonify(data_list):
    json_list = []
    for data in data_list:
        json_data = dict()
        for key, value in data.items():
            if isinstance(value, list):  # for lists
                value = [
                    jsonify(item) if isinstance(item, dict) else item
                    for item in value
                ]
            if isinstance(value, dict):  # for nested lists
                value = jsonify(value)
            if isinstance(key, int):  # if key is integer: > to string
                key = str(key)
            if type(
                    value
            ).__module__ == 'numpy':  # if value is numpy.*: > to python list
                value = value.tolist()
            json_data[key] = value
        json_list.append(json_data)
    return json_list


def create_result(result, threshold=0):
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

    main_json = create_results_json(labels_3d, scores_3d, bboxes_3d)

    return main_json


def create_results_json(labels, scores, bboxes):

    task_details = {"name": "semantic_slam:active:ground_truth", "results_format": "object_map"}
    environment_details = [{"name": "miniroom", "variant": 1}]
    main_json = {"task_details": task_details, "environment_details": environment_details}

    class_list = [
        'bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
        'night_stand', 'bookshelf', 'bathtub'
    ]

    results_json = create_empty()
    object_list = []
    for index, val in enumerate(labels):
        object = {
            "class": class_list[val],
            "class_ID": val,
            "confidence": scores[index].tolist(),
            "centroid": bboxes[index][:3].tolist(),
            "extent": bboxes[index][3:6].tolist(),
        }
        object_list.append(object)

    object_list = jsonify(object_list)

    for r in object_list:
        r['label_probs'] = [0] * len(class_list)
        if r['class'] in class_list:
            r['label_probs'][class_list.index(r['class'])] = r['confidence']

    results_json['objects'] = object_list
    results_json["class_list"] = class_list

    main_json["results"] = results_json

    return main_json


def vis_mm(model, data, result, threshold=0):

    points = data['inputs']['points']
    data_input = dict(points=points)
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
        pred_score_thr=threshold,
        vis_task='lidar_det')


def vis_mm_2(pcd, color=False, boxes=None):
    visualizer = Det3DLocalVisualizer(pcd_mode=2)
    if color:
        visualizer.set_points(pcd, mode='xyzrgb')
    else:
        visualizer.set_points(pcd, mode='xyz')
    if boxes is not None:
        visualizer.draw_bboxes_3d(boxes)
    visualizer.show()


def detect(model, pcd_path):
    result, data = inference_detector(model, pcd_path)
    vis_mm(model, data, result, threshold=0.7)
    return result
