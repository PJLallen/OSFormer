# -*- coding: utf-8 -*-
import logging
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from kornia.morphology import erosion

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n
from fvcore.nn import sigmoid_focal_loss_jit
from torchvision.transforms import transforms
from .utils import imrescale, center_of_mass, point_nms, mask_nms, matrix_nms
from .loss import dice_loss
from .trans_decoder import build_transformer_decoder
from .trans_encoder import build_transformer_encoder
from .position_encoding import build_position_encoding
from .instance_fusion import DCIN

__all__ = ["OSFormer"]


@META_ARCH_REGISTRY.register()
class OSFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.scale_ranges = cfg.MODEL.OSFormer.FEAT_SCALE_RANGES
        self.strides = cfg.MODEL.OSFormer.FEAT_INSTANCE_STRIDES
        self.sigma = cfg.MODEL.OSFormer.SIGMA
        self.num_classes = cfg.MODEL.OSFormer.NUM_CLASSES
        self.num_kernels = cfg.MODEL.OSFormer.NUM_KERNELS
        self.num_grids = cfg.MODEL.OSFormer.NUM_GRIDS

        self.instance_in_features = cfg.MODEL.OSFormer.INSTANCE_IN_FEATURES
        self.instance_strides = cfg.MODEL.OSFormer.FEAT_INSTANCE_STRIDES
        self.instance_in_channels = cfg.MODEL.OSFormer.INSTANCE_IN_CHANNELS
        self.instance_channels = cfg.MODEL.OSFormer.INSTANCE_CHANNELS

        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_in_features = cfg.MODEL.OSFormer.MASK_IN_FEATURES
        self.mask_in_channels = cfg.MODEL.OSFormer.MASK_IN_CHANNELS
        self.mask_channels = cfg.MODEL.OSFormer.MASK_CHANNELS
        self.num_masks = cfg.MODEL.OSFormer.NUM_MASKS

        self.resize_input_factor = cfg.MODEL.OSFormer.RESIZE_INPUT_FACTOR
        # self.confidence_score = cfg.MODEL.OSFormer.CONFIDENCE_SCORE
        self.max_before_nms = cfg.MODEL.OSFormer.NMS_PRE
        self.score_threshold = cfg.MODEL.OSFormer.SCORE_THR
        self.update_threshold = cfg.MODEL.OSFormer.UPDATE_THR
        self.mask_threshold = cfg.MODEL.OSFormer.MASK_THR
        self.max_per_img = cfg.MODEL.OSFormer.MAX_PER_IMG
        self.nms_kernel = cfg.MODEL.OSFormer.NMS_KERNEL
        self.nms_sigma = cfg.MODEL.OSFormer.NMS_SIGMA
        self.nms_type = cfg.MODEL.OSFormer.NMS_TYPE

        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()

        instance_shapes = [backbone_shape[f] for f in self.instance_in_features]
        self.cate_head = CISTransformerHead(cfg, instance_shapes)

        mask_shapes = [backbone_shape['res' + f[-1]] for f in self.mask_in_features]
        self.mask_head = C2FMaskHead(cfg, mask_shapes)

        self.ins_loss_weight = cfg.MODEL.OSFormer.LOSS.DICE_WEIGHT
        self.focal_loss_alpha = cfg.MODEL.OSFormer.LOSS.FOCAL_ALPHA
        self.focal_loss_gamma = cfg.MODEL.OSFormer.LOSS.FOCAL_GAMMA
        self.focal_loss_weight = cfg.MODEL.OSFormer.LOSS.FOCAL_WEIGHT

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        self.no_fpn = cfg.MODEL.OSFormer.NOFPN
        if self.no_fpn:
            self.res_modules = nn.ModuleList()
            for f in ['res2', 'res3', 'res4', 'res5']:
                in_chn = backbone_shape[f].channels
                self.res_modules.append(nn.Sequential(
                    nn.Conv2d(in_chn, self.mask_in_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.GroupNorm(32, self.mask_in_channels), nn.ReLU(inplace=True)))

        self.sem_loss_on = cfg.MODEL.OSFormer.SEM_LOSS
        self.sem_loss_weight = cfg.MODEL.OSFormer.LOSS.SEM_WEIGHT
        self.sem_loss_type = cfg.MODEL.OSFormer.LOSS.SEM_TYPE
        self.ins_edge_on = cfg.MODEL.OSFormer.INS_EDGE
        self.ins_edge_weight = cfg.MODEL.OSFormer.LOSS.INS_EDGE_WEIGHT
        self.ins_fusion = cfg.MODEL.OSFormer.INS_FUSION
        self.dcin = DCIN(self.num_kernels, cfg.MODEL.OSFormer.DCIN_NORM)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DetectionTransform` .
                Each item in the list contains the inputs for one image.
            For now, each item in the list is a dict that contains:
                image: Tensor, image in (C, H, W) format.
                instances: Instances
                Other information that's included in the original dicts, such as:
                    "height", "width" (int): the output resolution of the model, used in inference.
                        See :meth:`postprocess` for details.
         Returns:
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in
                            batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        if self.no_fpn:
            res_feats = ["res2", "res3", "res4", "res5"]
            features = {f: fn(features[f]) for f, fn in zip(res_feats, self.res_modules)}

        ins_features = [features[f] for f in self.instance_in_features]

        if len(self.instance_strides) > 3:
            ins_features = self.split_feats(ins_features)

        cate_pred, kernel_pred, mask_extra_feat = self.cate_head(ins_features)
        features.update({f_name.replace('res', 'trans'): feat
                         for f_name, feat in zip(self.instance_in_features, mask_extra_feat)})
        if features.get('trans2') is not None:
            features['trans2'] = F.interpolate(features['trans2'], scale_factor=2)

        mask_in_feats = [features[f] for f in self.mask_in_features]
        mask_pred = self.mask_head(mask_in_feats)

        sem_pred = None
        if self.sem_loss_on:
            mask_pred, sem_pred = mask_pred
        if self.training:
            """
            get_ground_truth.
            return loss and so on.
            """
            mask_feat_size = mask_pred.size()[-2:]
            sem_targets = None
            if self.sem_loss_on:
                sem_targets = self.get_sem_ground_truth(gt_instances, mask_feat_size)
            targets = self.get_ground_truth(gt_instances, mask_feat_size)

            losses = self.loss(cate_pred, kernel_pred, mask_pred, targets, sem_targets, sem_pred)
            return losses
        else:
            # point nms.
            cate_pred = [point_nms(cate_p.sigmoid(), kernel=2).permute(0, 2, 3, 1)
                         for cate_p in cate_pred]
            # do inference for results.
            results = self.inference(cate_pred, kernel_pred, mask_pred, images.image_sizes, batched_inputs, sem_pred)
            return results

    def map_to_edge(self, tensor):
        tensor = tensor.float()
        kernel = torch.ones((5, 5), device=tensor.device)
        ero_map = erosion(tensor, kernel)
        res = tensor - ero_map

        return res

    def Reize_the_input(self, image):
        c, h, w = list(image.size())
        target_h = int(self.resize_input_factor * h)
        target_w = int(self.resize_input_factor * w)
        transform = transforms.Resize((target_h, target_w))
        new_image = transform(image)
        return new_image

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]

        divisibility = self.backbone.size_divisibility
        if self.no_fpn:
            divisibility = 64
        images = ImageList.from_tensors(images, divisibility)
        return images

    @torch.no_grad()
    def get_sem_ground_truth(self, gt_instances, mask_feat_size):
        h, w = mask_feat_size
        gt_sem_list = []
        for img_idx in range(len(gt_instances)):
            gt_masks_raw = gt_instances[img_idx].gt_masks.tensor
            if gt_masks_raw.numel() == 0:
                continue
            output_stride = 4
            gt_masks_scale = F.interpolate(gt_masks_raw.float().unsqueeze(0), scale_factor=1. / output_stride,
                                           mode='nearest').squeeze(0)
            gt_masks_scale = gt_masks_scale.sum(dim=0)
            sem_target_pad = torch.zeros([h, w], dtype=torch.uint8, device=gt_masks_scale.device)
            sem_target_pad[:gt_masks_scale.shape[0], :gt_masks_scale.shape[1]] = gt_masks_scale

            gt_sem_list.append(sem_target_pad)

        if len(gt_sem_list) == 0:
            return torch.zeros((1, 1, h, w), dtype=torch.uint8, device=gt_instances.device)
        return torch.stack(gt_sem_list, dim=0).unsqueeze(1)

    @torch.no_grad()
    def get_ground_truth(self, gt_instances, mask_feat_size=None):
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = [], [], [], []
        for img_idx in range(len(gt_instances)):
            cur_ins_label_list, cur_cate_label_list, \
            cur_ins_ind_label_list, cur_grid_order_list = \
                self.get_ground_truth_single(img_idx, gt_instances, mask_feat_size=mask_feat_size)
            ins_label_list.append(cur_ins_label_list)
            cate_label_list.append(cur_cate_label_list)
            ins_ind_label_list.append(cur_ins_ind_label_list)
            grid_order_list.append(cur_grid_order_list)
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

    def get_ground_truth_single(self, img_idx, gt_instances, mask_feat_size):
        gt_bboxes_raw = gt_instances[img_idx].gt_boxes.tensor
        gt_labels_raw = gt_instances[img_idx].gt_classes
        gt_masks_raw = gt_instances[img_idx].gt_masks.tensor
        device = gt_labels_raw[0].device

        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        ins_label_list = []
        cate_label_list = []
        ins_ind_label_list = []
        grid_order_list = []
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.num_grids):

            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero(as_tuple=False).flatten()
            num_ins = len(hit_indices)

            ins_label = []
            grid_order = []
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            cate_label = torch.fill_(cate_label, self.num_classes)
            ins_ind_label = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)

            if num_ins == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                ins_label_list.append(ins_label)
                cate_label_list.append(cate_label)
                ins_ind_label_list.append(ins_ind_label)
                grid_order_list.append([])
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices, ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # mass center
            center_ws, center_hs = center_of_mass(gt_masks)
            valid_mask_flags = gt_masks.sum(dim=-1).sum(dim=-1) > 0

            output_stride = 4
            gt_masks = gt_masks.permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()
            gt_masks = imrescale(gt_masks, scale=1. / output_stride)
            if len(gt_masks.shape) == 2:
                gt_masks = gt_masks[..., None]
            gt_masks = torch.from_numpy(gt_masks).to(dtype=torch.uint8, device=device).permute(2, 0, 1)
            for seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in zip(gt_masks, gt_labels,
                                                                                               half_hs, half_ws,
                                                                                               center_hs, center_ws,
                                                                                               valid_mask_flags):
                if not valid_mask_flag:
                    continue
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                top = max(top_box, coord_h - 1)
                down = min(down_box, coord_h + 1)
                left = max(coord_w - 1, left_box)
                right = min(right_box, coord_w + 1)

                cate_label[top:(down + 1), left:(right + 1)] = gt_label
                for i in range(top, down + 1):
                    for j in range(left, right + 1):
                        label = int(i * num_grid + j)

                        cur_ins_label = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                    device=device)
                        cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        ins_label.append(cur_ins_label)
                        ins_ind_label[label] = True
                        grid_order.append(label)
            if len(ins_label) == 0:
                ins_label = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            else:
                ins_label = torch.stack(ins_label, 0)
            ins_label_list.append(ins_label)
            cate_label_list.append(cate_label)
            ins_ind_label_list.append(ins_ind_label)
            grid_order_list.append(grid_order)
        return ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list

    def loss(self, cate_preds, kernel_preds, ins_pred, targets, sem_targets=None, sem_pred=None):
        ins_label_list, cate_label_list, ins_ind_label_list, grid_order_list = targets
        # ins
        ins_labels = [torch.cat([ins_labels_level_img
                                 for ins_labels_level_img in ins_labels_level], 0)
                      for ins_labels_level in zip(*ins_label_list)]

        # TODO revise for top-k
        kernel_preds = [[kernel_preds_level_img.view(kernel_preds_level_img.shape[0], -1)[:, grid_orders_level_img]
                         for kernel_preds_level_img, grid_orders_level_img in
                         zip(kernel_preds_level, grid_orders_level)]
                        for kernel_preds_level, grid_orders_level in zip(kernel_preds, zip(*grid_order_list))]
        # generate masks
        ins_pred_list = []
        for b_kernel_pred in kernel_preds:
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred):

                if kernel_pred.size()[-1] == 0:
                    continue
                cur_ins_pred = ins_pred[idx, ...]
                H, W = cur_ins_pred.shape[-2:]
                cur_ins_pred = cur_ins_pred.unsqueeze(0)
                kernel_pred = kernel_pred.permute(1, 0)
                cur_ins_pred = self.dcin(cur_ins_pred, kernel_pred)

                cur_ins_pred = cur_ins_pred.view(-1, H, W)
                b_mask_pred.append(cur_ins_pred)
            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                b_mask_pred = torch.cat(b_mask_pred, 0)
            ins_pred_list.append(b_mask_pred)

        ins_ind_labels = [
            torch.cat([ins_ind_labels_level_img.flatten()
                       for ins_ind_labels_level_img in ins_ind_labels_level])
            for ins_ind_labels_level in zip(*ins_ind_label_list)
        ]
        flatten_ins_ind_labels = torch.cat(ins_ind_labels)

        num_ins = flatten_ins_ind_labels.sum()

        # dice loss
        loss_ins = []
        loss_ins_edge = []
        for input, target in zip(ins_pred_list, ins_labels):
            if input is None:
                continue
            input = torch.sigmoid(input)
            loss_ins.append(dice_loss(input, target))
            if self.ins_edge_on:
                input_edge = self.map_to_edge(input.unsqueeze(0)).squeeze(0)
                target_edge = self.map_to_edge(target.unsqueeze(0)).squeeze(0)
                loss_ins_edge.append(dice_loss(input_edge, target_edge))

        loss_ins_mean = torch.cat(loss_ins).mean()
        loss_ins = loss_ins_mean * self.ins_loss_weight
        loss_ins_edge = torch.cat(loss_ins_edge).mean() * self.ins_edge_weight if self.ins_edge_on else []

        # cate
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)

        # prepare one_hot
        pos_inds = torch.nonzero(flatten_cate_labels != self.num_classes).squeeze(1)

        flatten_cate_labels_oh = torch.zeros_like(flatten_cate_preds)
        flatten_cate_labels_oh[pos_inds, flatten_cate_labels[pos_inds]] = 1

        loss_cate = self.focal_loss_weight * sigmoid_focal_loss_jit(flatten_cate_preds, flatten_cate_labels_oh,
                                                                    gamma=self.focal_loss_gamma,
                                                                    alpha=self.focal_loss_alpha,
                                                                    reduction="sum") / (num_ins + 1)

        if self.sem_loss_on:
            sem_targets = self.map_to_edge(sem_targets)
            if not isinstance(sem_pred, list):
                sem_preds = [sem_pred]
            else:
                sem_preds = sem_pred

            loss_sem = 0
            for sem_pred in sem_preds:
                if self.sem_loss_type == 'focal':
                    num_pos = (sem_targets > 0).sum().float().clamp(min=1.0)
                    loss_sem += sigmoid_focal_loss_jit(
                        sem_pred, sem_targets.float(), gamma=self.focal_loss_gamma,
                        alpha=self.focal_loss_alpha, reduction="sum") / num_pos
                elif self.sem_loss_type == 'bce':
                    loss_sem += F.binary_cross_entropy_with_logits(sem_pred, sem_targets.float())
                else:
                    sem_pred = F.interpolate(sem_pred.sigmoid(), sem_targets.shape[-2:])
                    loss_sem += dice_loss(sem_pred, sem_targets).mean()

            return {'loss_ins': loss_ins, 'loss_cate': loss_cate, 'loss_sem': loss_sem * self.sem_loss_weight}

        if self.ins_edge_on:
            return {'loss_ins': loss_ins, 'loss_cate': loss_cate, 'loss_ins_edge': loss_ins_edge}

        return {'loss_ins': loss_ins, 'loss_cate': loss_cate}

    @staticmethod
    def split_feats(feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                feats[1],
                feats[2],
                feats[3])

    def inference(self, pred_cates, pred_kernels, pred_masks, cur_sizes, images, pred_sems=None):
        assert len(pred_cates) == len(pred_kernels)

        results = []
        num_ins_levels = len(pred_cates)
        for img_idx in range(len(images)):
            # image size.
            ori_img = images[img_idx]
            height, width = ori_img["height"], ori_img["width"]
            ori_size = (height, width)

            # prediction.
            pred_cate = [pred_cates[i][img_idx].view(-1, self.num_classes).detach()
                         for i in range(num_ins_levels)]
            pred_kernel = [pred_kernels[i][img_idx].permute(1, 2, 0).view(-1, self.num_kernels).detach()
                           for i in range(num_ins_levels)]
            pred_mask = pred_masks[img_idx, ...].unsqueeze(0)

            pred_cate = torch.cat(pred_cate, dim=0)
            pred_kernel = torch.cat(pred_kernel, dim=0)

            # inference for single image.
            result = self.inference_single_image(pred_cate, pred_kernel, pred_mask,
                                                 cur_sizes[img_idx], ori_size)
            if self.sem_loss_on:
                results.append({"pred_sems": pred_sems[-1][img_idx], "instances": result})
            else:
                results.append({"instances": result})
        return results

    def inference_single_image(
            self, cate_preds, kernel_preds, seg_preds, cur_size, ori_size
    ):
        # cate_preds = torch.ones_like(cate_preds).to(cate_preds.device)
        # overall info.
        h, w = cur_size
        f_h, f_w = seg_preds.size()[-2:]
        ratio = math.ceil(h / f_h)
        upsampled_size_out = (int(f_h * ratio), int(f_w * ratio))

        # process.
        inds = (cate_preds > self.score_threshold)
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])

            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        size_trans = cate_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(self.num_grids)
        strides[:size_trans[0]] *= self.instance_strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.instance_strides[ind_]
        strides = strides[inds[:, 0]]

        # mask encoding.
        seg_preds = self.dcin(seg_preds, kernel_preds).sigmoid()
        seg_preds = seg_preds.squeeze(0)

        # mask.
        seg_masks = seg_preds > self.mask_threshold
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # mask scoring.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.max_before_nms:
            sort_inds = sort_inds[:self.max_before_nms]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        if self.nms_type == "matrix":
            # matrix nms & filter.
            cate_scores = matrix_nms(cate_labels, seg_masks, sum_masks, cate_scores,
                                     sigma=self.nms_sigma, kernel=self.nms_kernel)
            keep = cate_scores >= self.update_threshold
        elif self.nms_type == "mask":
            # original mask nms.
            keep = mask_nms(cate_labels, seg_masks, sum_masks, cate_scores,
                            nms_thr=self.mask_threshold)
        else:
            raise NotImplementedError

        if keep.sum() == 0:
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.max_per_img:
            sort_inds = sort_inds[:self.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # reshape to original size.
        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                                  size=ori_size,
                                  mode='bilinear').squeeze(0)
        seg_preds_cp = seg_masks
        seg_masks = seg_masks > self.mask_threshold

        results = Instances(ori_size)
        results.pred_classes = cate_labels
        results.scores = cate_scores
        results.pred_masks = seg_masks
        results.seg_preds = seg_preds_cp

        # get bbox from mask
        pred_boxes = torch.zeros(seg_masks.size(0), 4)

        results.pred_boxes = Boxes(pred_boxes)

        return results


class CISTransformerHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        OSFormer Cate Head.
        """
        super().__init__()

        self.num_classes = cfg.MODEL.OSFormer.NUM_CLASSES
        self.num_kernels = cfg.MODEL.OSFormer.NUM_KERNELS
        self.num_grids = cfg.MODEL.OSFormer.NUM_GRIDS
        self.instance_in_features = cfg.MODEL.OSFormer.INSTANCE_IN_FEATURES
        self.instance_strides = cfg.MODEL.OSFormer.FEAT_INSTANCE_STRIDES
        self.instance_in_channels = cfg.MODEL.OSFormer.INSTANCE_IN_CHANNELS
        self.instance_channels = cfg.MODEL.OSFormer.INSTANCE_CHANNELS
        self.num_levels = len(self.instance_in_features)
        self.hidden_dim = cfg.MODEL.OSFormer.HIDDEN_DIM
        self.no_fpn = cfg.MODEL.OSFormer.NOFPN
        if self.no_fpn:
            in_channels = [256 for _ in input_shape]
        else:
            in_channels = [s.channels for s in input_shape]

        assert len(set(in_channels)) == 1, \
            print("Each level must have the same channel!")
        in_channels = in_channels[0]
        assert in_channels == cfg.MODEL.OSFormer.INSTANCE_IN_CHANNELS, \
            print("In channels should equal to tower in channels!")

        self.trans_encoder = build_transformer_encoder(cfg)
        self.trans_decoder = build_transformer_decoder(cfg)

        self.cate_pred = nn.Linear(self.instance_in_channels, self.num_classes)
        self.kernel_pred = nn.Linear(self.instance_in_channels, self.num_kernels)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.OSFormer.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cate_pred.bias, bias_value)

        # query selection
        self.qs_enable = cfg.MODEL.OSFormer.QS.ENABLE
        if self.qs_enable:
            self.qs_input = cfg.MODEL.OSFormer.QS.INPUT
            self.qs_share_head = cfg.MODEL.OSFormer.QS.SHARE_HEAD
            self.qs_nums = cfg.MODEL.OSFormer.QS.NUM_QUERIES
            if self.qs_share_head:
                self.qs_pred = self.cate_pred
            else:
                self.qs_pred = nn.Linear(self.instance_in_channels, self.num_kernels)
                torch.nn.init.constant_(self.qs_pred.bias, bias_value)

        self.position_encoding = build_position_encoding(self.hidden_dim)

    def query_select(self, feat_list, pos_list):
        """
        select query direct from memory sequence
        """
        if not self.qs_enable:
            return

        feat_seq = torch.cat(feat_list, 1)
        pos_seq = torch.cat(pos_list, 1)
        output_cate = self.qs_pred(feat_seq)
        topk = self.qs_nums
        topk_feats, topk_indices = torch.topk(output_cate[..., 0], topk, dim=1)
        topk_pos = torch.gather(pos_seq, 1, topk_indices)

        return topk_feats, topk_pos

    def forward(self, features):

        pos_encoder = []
        sizes_encoder = []
        for idx, feature in enumerate(features):
            position_embedding = self.position_encoding(feature)
            pos_encoder.append(position_embedding)
            bs, _, w, h = feature.shape
            sizes_encoder.append((w, h))

        split_list = [w * h for (w, h) in sizes_encoder]
        trans_memory = []
        memorys, level_start_index = self.trans_encoder(features, pos_encoder)

        pos_grid = []
        sizes_decoder = []
        srcs_decoder = []
        for memory, (w, h), seg_num_grid in zip(memorys.split(split_list, 1), sizes_encoder, self.num_grids):
            memory = memory.view((bs, w, h, -1)).permute(0, 3, 1, 2)
            trans_memory.append(memory)
            ins_kernel_feat = memory
            feat = F.interpolate(ins_kernel_feat, size=seg_num_grid, mode='bilinear')  # (bs, c, w, h)
            pos_grid.append(self.position_encoding(feat))
            w, h = feat.shape[-2:]
            sizes_decoder.append((w, h))
            srcs_decoder.append(feat)

        cate_pred = []
        kernel_pred = []
        hss, level_start_index = self.trans_decoder(srcs_decoder, pos_grid,
                                                                            trans_memory, pos_encoder)
        split_list = [w * h for (w, h) in sizes_decoder]
        for memory, (w, h) in zip(hss.split(split_list, 1), sizes_decoder):
            hs = memory.view((bs, w, h, -1))
            cate_pred_single = self.cate_pred(hs).permute(0, 3, 1, 2)
            cate_pred.append(cate_pred_single)
            kernel_pred_single = self.kernel_pred(hs).permute(0, 3, 1, 2)
            kernel_pred.append(kernel_pred_single)

        return cate_pred, kernel_pred, trans_memory


class C2FMaskHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Coarse to fine Mask Head.
        """
        super().__init__()

        self.mask_on = cfg.MODEL.MASK_ON
        self.num_masks = cfg.MODEL.OSFormer.NUM_MASKS
        self.mask_in_features = cfg.MODEL.OSFormer.MASK_IN_FEATURES
        self.mask_in_channels = cfg.MODEL.OSFormer.MASK_IN_CHANNELS
        self.mask_channels = cfg.MODEL.OSFormer.MASK_CHANNELS
        self.num_levels = len(input_shape)
        assert self.num_levels == len(self.mask_in_features), \
            print("Input shape should match the features.")

        norm = None if cfg.MODEL.OSFormer.NORM == "none" else cfg.MODEL.OSFormer.NORM

        self.convs_all_levels = nn.ModuleList()
        self.convs_all_sums = nn.ModuleList()
        for i in range(self.num_levels):
            convs_per_level = nn.Sequential()

            conv_tower = list()
            in_chn = self.mask_channels
            if i == self.num_levels - 1:
                in_chn = self.mask_in_channels
            conv_tower.append(nn.Conv2d(
                in_chn, self.mask_channels,
                kernel_size=3, stride=1,
                padding=1, bias=norm is None
            ))
            if norm == "GN":
                conv_tower.append(nn.GroupNorm(32, self.mask_channels))

            convs_per_level.add_module('conv' + str(i), nn.Sequential(*conv_tower))
            if i != 0:
                upsample_tower = nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=False)
                convs_per_level.add_module(
                    'upsample' + str(i), upsample_tower)
            self.convs_all_sums.append(convs_per_level)

            if i == self.num_levels - 1:
                continue
            channel_conv_tower = nn.Sequential(
                nn.Conv2d(
                    self.mask_in_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None),
                nn.GroupNorm(32, self.mask_channels)
            )
            self.convs_all_levels.append(channel_conv_tower)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                self.mask_channels, self.num_masks,
                kernel_size=1, stride=1,
                padding=0, bias=norm is None),
            nn.GroupNorm(32, self.num_masks),
            nn.ReLU(inplace=True)
        )

        for modules in [self.convs_all_levels, self.conv_pred, self.convs_all_sums]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

        self.sem_loss_on = cfg.MODEL.OSFormer.SEM_LOSS
        self.single_sem = cfg.MODEL.OSFormer.SINGLE_SEM
        if self.sem_loss_on:
            self.edge_all_levels = nn.ModuleList()
            if self.single_sem:
                self.edge_all_levels.append(ReverseEdgeSupervision(self.mask_channels))
            else:
                for _ in range(self.num_levels - 1):
                    self.edge_all_levels.append(ReverseEdgeSupervision(self.mask_channels))

    def forward(self, features):
        assert len(features) == self.num_levels, \
            print("The number of input features should be equal to the supposed level.")

        mask_feat = features[-1]

        feature_add_all_level = self.convs_all_sums[-1](mask_feat)
        edge_preds = []
        feat_pre_level = None
        for i in range(self.num_levels - 2, -1, -1):
            feat_pre_level = self.convs_all_levels[i](features[i])
            feature_add_all_level += feat_pre_level
            feature_add_all_level = F.relu(feature_add_all_level, inplace=True)
            if self.sem_loss_on and not self.single_sem:
                edge_preds.append(self.edge_all_levels[2 - i](
                    feature_add_all_level, feat_pre_level))
            feature_add_all_level = self.convs_all_sums[i](feature_add_all_level)

        feature_add_all_level = F.relu(feature_add_all_level, inplace=True)
        if self.sem_loss_on and self.single_sem:
            edge_preds.append(self.edge_all_levels[0](
                feature_add_all_level, feat_pre_level))
        mask_pred = self.conv_pred(feature_add_all_level)

        if self.sem_loss_on:
            return mask_pred, edge_preds

        return mask_pred


class ReverseEdgeSupervision(nn.Module):
    def __init__(self, chn):
        super().__init__()

        self.edge_pred = nn.Conv2d(
            chn, 1,
            kernel_size=3, stride=1,
            padding=1, bias=False)
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, feat_fuse, feat_high):
        avg_high = torch.mean(feat_high, dim=1, keepdim=True)
        max_high, _ = torch.max(feat_high, dim=1, keepdim=True)
        x = torch.cat([avg_high, max_high], dim=1)
        x = 1 - self.conv1(x).sigmoid()

        fuse = feat_fuse * x
        return self.edge_pred(fuse)
