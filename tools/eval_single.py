import os

import tqdm
import json
from collections import OrderedDict
from detectron2.engine.defaults import DefaultPredictor
from detectron2.evaluation.coco_evaluation import COCOEvaluator

from adet.config import get_cfg
from adet.data.datasets.cis import register_dataset
from tools.train_net import Trainer


def setup_cfg(config_file, model_weights, confidence_threshold):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    # Set score_threshold for builtin models
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()

    return cfg


def main(config_file, model_weights, dataset_name, output_dir=None, confidence_threshold=0.3):
    cfg = setup_cfg(config_file, model_weights, confidence_threshold)

    if output_dir is None:
        output_dir = os.path.dirname(model_weights)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    predictor = DefaultPredictor(cfg)
    model = predictor.model
    data_loader = Trainer.build_test_loader(cfg, dataset_name)
    coco_eval = COCOEvaluator(dataset_name, output_dir=output_dir, tasks=('segm',))

    eval_res = OrderedDict()
    for elem in tqdm.tqdm(data_loader):
        predictions = model(elem)
        coco_eval.reset()
        coco_eval.process(elem, predictions)
        eval_cur = coco_eval.evaluate()
        filename = os.path.basename(elem[0]['file_name'])
        eval_res[filename] = eval_cur['segm']

    with open(os.path.join(output_dir, '{}_ap.json'.format(dataset_name)), 'w') as f:
        json.dump(eval_res, f, indent=4)
