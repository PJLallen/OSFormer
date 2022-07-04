import os
import cv2
import json
from collections import defaultdict, OrderedDict

import torch
import pycocotools.mask as mask_util
from pycocotools.coco import COCO
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from adet.data.datasets.cis import register_dataset


def pre_process_json(json_file, score_threshold=0.3):
    with open(json_file, 'r') as f:
        data = json.load(f)

    data_filtered = defaultdict(list)
    for d in data:
        if d['score'] > score_threshold:
            data_filtered[d['image_id']].append(d)

    return data_filtered


def data2instance(data):
    instances = {}
    for k in data.keys():
        if len(data[k]) == 0:
            instances[k] = None
        results = Instances(data[k][0]['segmentation']['size'])
        scores = []
        segms = []
        for elem in data[k]:
            scores.append(elem['score'])
            segms.append(mask_util.decode(elem['segmentation']))
        results.scores = torch.as_tensor(scores)
        results.pred_masks = torch.as_tensor(segms)
        instances[k] = results

    return instances


def vis_single_image(img_info, ins, img_dir, output_dir, instance_mode=ColorMode.IMAGE):
    img_filename = img_info['file_name']
    if ins is None:
        ins = Instances((img_info['height'], img_info['width']))

    im = cv2.imread(os.path.join(img_dir, img_filename))[:, :, ::-1]
    visualizer = Visualizer(im, instance_mode=instance_mode)
    vis_output = visualizer.draw_instance_predictions(predictions=ins)
    out_filename = os.path.join(output_dir, img_filename)
    print('Save {} successfully.'.format(out_filename))
    vis_output.save(out_filename)


def eval_single_image(coco_eval, prediction):
    coco_eval.reset()
    coco_eval._predictions.append(prediction)
    return coco_eval.evaluate()


def vis(res_json, dataset_name, output_dir=None, score_threshold=0.3):
    if output_dir is None:
        output_dir = os.path.dirname(res_json)

    vis_dir = os.path.join(output_dir, 'vis')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    anno_json, img_dir = datasets[dataset_name]
    coco = COCO(anno_json)
    coco_eval = COCOEvaluator(dataset_name, output_dir=output_dir, tasks=('segm',))

    data = pre_process_json(res_json, score_threshold)
    instances = data2instance(data)
    print('Get instances successfully.')

    eval_res = OrderedDict()
    for img_id, ins in instances.items():
        vis_single_image(coco.imgs[img_id], ins, img_dir, vis_dir)
        prediction = {"image_id": img_id, "instances": data[img_id]}
        eval_cur = eval_single_image(coco_eval, prediction)
        eval_res[coco.imgs[img_id]['file_name']] = eval_cur['segm']

    with open(os.path.join(output_dir, '{}_ap.json'.format(dataset_name)), 'w') as f:
        json.dump(eval_res, f, indent=4)


if __name__ == '__main__':
    datasets = {
        "my_data_test_coco_cod_style": [
            'COD10K-v3/annotations/test2026.json',
            'COD10K-v3/Test/Image/'
        ],
        "my_data_test_coco_nc4k_style": [
            'NC4K/nc4k_test.json',
            'NC4K/Imgs/'
        ]
    }

    register_dataset()
