from typing import Optional, Tuple, List, Dict

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import RoIPool, box_iou, batched_nms, clip_boxes_to_image, remove_small_boxes, RoIAlign

from utils.utils import ProposalMathcer, BalancedBatchSampler, boxes_to_transforms, transforms_to_boxes


Tensor = torch.Tensor


class FastRCNN(nn.Module):
    def __init__(
            self,
            mask_branch: bool,
            roi_output_size: int or Tuple[int, int],
            roi_spatial_scale: float,
            backbone_out_channels: int,
            nms_iou_pos_threshold: float,
            nms_iou_neg_threshold: float,
            image_size: Tuple[int, int],
            representation_dim: int,
            num_classes: int,
            fast_rcnn_batch_size: int,
            inference_score_thresh: float,
            inference_nms_thresh: float,
            post_nms_top_n: Dict[str, int],
            detections_per_img: int
    ):
        super(FastRCNN, self).__init__()

        self.mask_branch = mask_branch

        self.image_size = image_size
        self.num_classes = num_classes
        self.fast_rcnn_batch_size = fast_rcnn_batch_size
        self.inference_score_thresh = inference_score_thresh
        self.inference_nms_thresh = inference_nms_thresh
        self.post_nms_top_n = post_nms_top_n
        self.detections_per_img = detections_per_img

        self.proposal_matcher = ProposalMathcer(
            nms_iou_pos_threshold, nms_iou_neg_threshold, image_size
        )

        assert fast_rcnn_batch_size % 2 == 0
        self.batch_sampler = BalancedBatchSampler(fast_rcnn_batch_size)

        self.roi = RoIPool(roi_output_size, roi_spatial_scale)

        roi_box_size = self.roi.output_size[0]
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out_channels * roi_box_size ** 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, representation_dim),
            nn.ReLU()
        )

        self.box_predictor = nn.Linear(representation_dim, 4 * num_classes)
        self.cls_predictor = nn.Linear(representation_dim, num_classes)

        self.box_criterion = nn.SmoothL1Loss(reduction='sum')
        self.cls_criterion = nn.CrossEntropyLoss(reduction='mean')

        if self.mask_branch:
            self.mask_roi_pool = RoIAlign(
                output_size=14,
                spatial_scale=roi_spatial_scale * 2,
                sampling_ratio=2)

            head_layers = []
            in_features = backbone_out_channels
            for _ in range(4):
                head_layers.append(
                    nn.Conv2d(in_features, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
                )
                head_layers.append(
                    nn.ReLU()
                )
                in_features = 256
            self.mask_head = nn.Sequential(*head_layers)

            self.mask_predictor = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
            )

    def forward(
            self,
            backbone_features: Tensor,
            proposals: List[Tensor],
            gt_boxes: Optional[List[Tensor]] = None,
            gt_classes: Optional[List[Tensor]] = None,
            gt_masks: Optional[List[Tensor]] = None
    ) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]:

        if self.training:
            matched_gt_boxes, gt_labels = self._assign_targets_to_proposals(proposals, gt_boxes, gt_classes)
            gt_transforms = boxes_to_transforms(matched_gt_boxes, proposals)

            proposals, gt_transforms, gt_labels = \
                self._collate_fn(proposals, gt_transforms, gt_labels)

        else:
            matched_gt_boxes, gt_labels, gt_transforms = None, None, None

        rois = self.roi(backbone_features, proposals)  # [K, C, h, w]
        features = self.mlp(rois)
        pred_transforms = self.box_predictor(features)
        pred_logits = self.cls_predictor(features)

        loss = {}
        output = []
        if self.training:
            loss = self.compute_loss(
                pred_transforms, pred_logits, gt_transforms, gt_labels
            )
        else:
            boxes, scores, labels = self._postprocess_predictions(pred_transforms, pred_logits, proposals)

            for i in range(len(boxes)):
                output.append(
                    {
                        'boxes': boxes[i],
                        'scores': scores[i],
                        'labels': labels[i]
                    }
                )

        if self.mask_branch:
            if self.training:
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_gt_boxes = []
                for img_id in range(num_images):
                    pos = torch.where(gt_labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_gt_boxes.append(matched_gt_boxes[img_id][pos])
            else:
                mask_proposals = [o['boxes'] for o in output]
                pos_matched_idxs = None

            mask_features = self.mask_roi_pool(backbone_features, mask_proposals)
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)

            if self.training:
                loss_mask = self.compute_mask_loss()
                loss['loss'] += loss_mask['loss_mask']
                loss['loss_mask'] = loss_mask['loss_mask'].detach()
            else:
                labels = [o['labels'] for o in output]
                masks_probs = self._postprocess_masks(mask_logits, labels)
                for mask_prob, o in zip(masks_probs, output):
                    o['masks'] = mask_prob

        return output, loss

    def compute_loss(
            self,
            pred_transforms: Tensor,
            pred_logits: Tensor,
            gt_transforms: List[Tensor],
            gt_labels: List[Tensor]
    ) -> Dict[str, Tensor]:

        gt_transforms = torch.cat(gt_transforms, dim=0)
        gt_labels = torch.cat(gt_labels, dim=0)

        cls_loss = self.cls_criterion(pred_logits, gt_labels)

        num_boxes = pred_transforms.shape[0]
        pred_transforms = pred_transforms.reshape(num_boxes, pred_transforms.shape[-1] // 4, 4)
        pos_idxs = torch.where(gt_labels > 0)[0]
        pos_labels = gt_labels[pos_idxs]

        box_loss = self.box_criterion(pred_transforms[pos_idxs, pos_labels], gt_transforms[pos_idxs]) / gt_labels.numel()

        loss = box_loss + cls_loss

        return {'loss': loss, 'box_loss': box_loss.detach(), 'cls_loss': cls_loss.detach()}

    def compute_mask_loss(self):
        return {'loss_mask': torch.tensor(0)}

    def _assign_targets_to_proposals(
            self,
            proposals: List[Tensor],
            gt_boxes: List[Tensor],
            gt_classes: List[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor]]:

        batch_sise = len(proposals)

        matched_boxes = []
        labels = []

        for i in range(batch_sise):
            proposals_i = proposals[i]
            gt_boxes_i, gt_classes_i = gt_boxes[i], gt_classes[i]

            matched_idxs_i = self.proposal_matcher(proposals_i, gt_boxes_i)
            clamped_matched_idxs_i = torch.clamp(matched_idxs_i, min=0)

            matched_boxes_i = gt_boxes_i[clamped_matched_idxs_i]
            classes_i = gt_classes_i[clamped_matched_idxs_i]

            neg_idxs = matched_idxs_i == self.proposal_matcher.NEG_CONST
            classes_i[neg_idxs] = 0

            discard_idxs = matched_idxs_i == self.proposal_matcher.DISCARD_CONST
            classes_i[discard_idxs] = -1

            matched_boxes.append(matched_boxes_i)
            labels.append(classes_i)

        return matched_boxes, labels

    def _collate_fn(
            self,
            proposals: List[Tensor],
            gt_transforms: List[Tensor],
            gt_labels: List[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:

        batch_size = len(proposals)

        batched_proposals = []
        batched_gt_transforms = []
        batched_gt_labels = []

        for i in range(batch_size):
            proposals_i = proposals[i]
            gt_transforms_i, gt_labels_i = gt_transforms[i], gt_labels[i]

            pos_idxs, neg_idxs = self.batch_sampler(gt_labels_i)

            batched_proposals.append(
                torch.cat([proposals_i[pos_idxs], proposals_i[neg_idxs]], dim=0)
            )

            batched_gt_transforms.append(
                torch.cat([gt_transforms_i[pos_idxs], gt_transforms_i[neg_idxs]], dim=0)
            )

            batched_gt_labels.append(
                torch.cat([gt_labels_i[pos_idxs], gt_labels_i[neg_idxs]], dim=0)
            )

        return batched_proposals, batched_gt_transforms, batched_gt_labels

    def _postprocess_predictions(
            self,
            pred_transforms: Tensor,
            pred_logits: Tensor,
            proposals: List[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]

        pred_boxes = transforms_to_boxes(pred_transforms, proposals)
        pred_scores = F.softmax(pred_logits, dim=-1)

        pred_boxes_list = torch.split(pred_boxes, boxes_per_image)
        pred_scores_list = torch.split(pred_scores, boxes_per_image)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores in zip(pred_boxes_list, pred_scores_list):
            boxes = clip_boxes_to_image(boxes, self.image_size)

            labels = torch.arange(self.num_classes, device=pred_transforms.device)
            labels = labels.view(1, -1).expand_as(pred_scores)

            boxes, scores, labels = boxes[:, 1:], scores[:, 1:], labels[:, 1:]

            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            keep = torch.where(scores > self.inference_score_thresh)[0]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            keep = remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            keep = batched_nms(boxes, scores, labels, self.inference_nms_thresh)
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def _post_nms_top_n(self) -> int:
        if self.training:
            return self.post_nms_top_n['training']
        return self.post_nms_top_n['testing']

    @staticmethod
    def _postprocess_masks(x: Tensor, labels: List[Tensor]) -> List[Tensor]:
        mask_prob = x.sigmoid()

        num_masks = x.shape[0]
        boxes_per_image = [label.shape[0] for label in labels]
        labels = torch.cat(labels)
        index = torch.arange(num_masks, device=labels.device)
        mask_prob = mask_prob[index, labels][:, None]
        mask_prob = mask_prob.split(boxes_per_image, dim=0)

        return mask_prob


