# made by EFCK - Custom data adaptation (ported from SymPointV2):
#   1. EvaluatorConfig.ignore_label: changed from single int to list[int] for multi-class ignore
#   2. Evaluator.eval_panoptic_quality(): all == ignore_label comparisons changed to torch.isin()
#   3. MetricsComputer: added ignore_label support to _compute_panoptic_quality() and
#      _compute_f1_scores() — ignored classes are excluded from aggregate PQ/F1 metrics

from typing import Mapping, Dict, List, Optional
from dataclasses import dataclass
import os
import json

import torch
from transformers.trainer import EvalPrediction


@dataclass
class EvaluatorConfig:
    num_classes: int
    ignore_label: list[int]
    iou_threshold: float
    output_dir: str


class Evaluator:
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self.num_classes = config.num_classes
        self.ignore_label = config.ignore_label  # list[int]
        self.iou_threshold = config.iou_threshold
        self.output_dir = config.output_dir

    def __call__(self, preds, targets):
        return self.eval_panoptic_quality(preds, targets), self.eval_semantic_quality(
            preds["pred_sem_segs"], targets["sem_labels"], targets["prim_lens"]
        )

    def eval_panoptic_quality(self, preds, targets):
        """
        Calculate panoptic quality metrics: PQ, SQ, RQ

        NOTE:a list of tensors means a batch of data, each tensor represents a sample in the batch

        Args:

            `preds` (`Dict[str, List[torch.Tensor]]`): In panoptic quality, the preds should have the following keys:

                \\- `pred_masks` (`List[torch.Tensor]`):
                    a list of tensors, each tensor shape is (num_predictions, num_primitives)
                    each line represents a predicted instance, each column represents a primitive,
                    each value is 0 or 1, 1 means the primitive is part of the instance

                \\- `pred_labels` (`List[torch.Tensor]`):
                    a list of tensors, each tensor shape is (num_predictions,)
                    each value represents the class of the predicted instance

            `targets` (`Dict[str, List[torch.Tensor]]`): In panoptic quality, the targets should have the following keys:

                \\- `target_masks` (`List[torch.Tensor]`):
                    a list of tensors, each tensor shape is (num_ground_truths, num_primitives)
                    each line represents a ground truth instance, each column represents a primitive,
                    each value is 0 or 1, 1 means the primitive is part of the instance

                \\- `target_labels` (`List[torch.Tensor]`):
                    a list of tensors, each tensor shape is (num_ground_truths,)
                    each value represents the class of the ground truth instance

                \\- `prim_lens` (`List[torch.Tensor]`):
                    a list of tensors, each tensor shape is (num_primitives,)
                    each value represents the length of the primitive

        Returns:

            `metric_states` (`Dict[str, torch.Tensor]`): Dictionary containing intermediate states for metric calculation

                \\- `tp_per_class` (`torch.Tensor`, shape is (num_classes,)): Number of true positive instances for each class

                \\- `fp_per_class` (`torch.Tensor`, shape is (num_classes,)): Number of false positive instances for each class

                \\- `fn_per_class` (`torch.Tensor`, shape is (num_classes,)): Number of false negative instances for each class

                \\- `tp_iou_score_per_class` (`torch.Tensor`, shape is (num_classes,)): Summary of True positive IoU score for each class
        """
        # Initialize the metric states
        tp_per_class = torch.zeros(
            self.num_classes, dtype=torch.int32, device=preds["pred_masks"][0].device
        )
        fp_per_class = torch.zeros(
            self.num_classes, dtype=torch.int32, device=preds["pred_masks"][0].device
        )
        fn_per_class = torch.zeros(
            self.num_classes, dtype=torch.int32, device=preds["pred_masks"][0].device
        )
        tp_iou_score_per_class = torch.zeros(
            self.num_classes, dtype=torch.float32, device=preds["pred_masks"][0].device
        )
        # Log lengths to degrade the influence of lines with very large span. (Follow the FloorPlanCAD paper)
        log_prim_lens = [torch.log(1 + prim_len) for prim_len in targets["prim_lens"]]
        # Iterate over all batches
        for batch_idx in range(len(preds["pred_masks"])):
            pred_masks = preds["pred_masks"][batch_idx]  # (num_preds, num_primitives)
            pred_labels = preds["pred_labels"][batch_idx]  # (num_preds,)
            target_masks = targets["target_masks"][batch_idx]  # (num_targets, num_primitives)
            target_labels = targets["target_labels"][batch_idx]  # (num_targets,)
            prim_lens = log_prim_lens[batch_idx]  # (num_primitives,)

            # Pre-compute ignore label tensor for isin checks
            ignore_labels_t = torch.tensor(self.ignore_label, device=target_labels.device)
            
            # changed by efck: vectorized IoU computation to avoid O(n^2) Python loop
            # Skip if no predictions or targets
            if pred_masks.shape[0] == 0 or target_masks.shape[0] == 0:
                # Count all targets as false negatives
                for target_idx in range(target_masks.shape[0]):
                    target_label = target_labels[target_idx]
                    if not torch.isin(target_label, ignore_labels_t):
                        fn_per_class[target_label] += 1
                continue
            
            # Filter out ignored labels
            valid_pred_mask = ~torch.isin(pred_labels, ignore_labels_t)
            valid_target_mask = ~torch.isin(target_labels, ignore_labels_t)
            
            if not valid_pred_mask.any() or not valid_target_mask.any():
                # Count valid targets as false negatives
                for target_idx in range(target_masks.shape[0]):
                    target_label = target_labels[target_idx]
                    if not torch.isin(target_label, ignore_labels_t):
                        fn_per_class[target_label] += 1
                continue
            
            # Compute IoU matrix in a vectorized manner: (num_targets, num_preds)
            # changed by efck: compute full IoU matrix at once instead of per-pair
            iou_matrix = self._calculate_iou_matrix(pred_masks, target_masks, prim_lens)
            
            # For each target, find predictions with IoU > threshold
            for target_idx in range(target_masks.shape[0]):
                target_label = target_labels[target_idx]
                if torch.isin(target_label, ignore_labels_t):
                    continue
                
                # Get IoU scores for this target with all predictions
                target_ious = iou_matrix[target_idx]  # (num_preds,)
                
                # Find predictions above threshold
                above_threshold = target_ious > self.iou_threshold
                
                if not above_threshold.any():
                    fn_per_class[target_label] += 1
                else:
                    # Check matching predictions
                    matching_pred_indices = torch.where(above_threshold)[0]
                    found_match = False
                    for pred_idx in matching_pred_indices:
                        pred_label = pred_labels[pred_idx]
                        if torch.isin(pred_label, ignore_labels_t):
                            continue
                        iou = target_ious[pred_idx]
                        if pred_label == target_label:
                            tp_per_class[pred_label] += 1
                            tp_iou_score_per_class[pred_label] += iou
                            found_match = True
                        else:
                            fp_per_class[pred_label] += 1
                    if not found_match:
                        fn_per_class[target_label] += 1
                        
        return dict(
            tp_per_class=tp_per_class,
            fp_per_class=fp_per_class,
            fn_per_class=fn_per_class,
            tp_iou_score_per_class=tp_iou_score_per_class,
        )
    
    def _calculate_iou_matrix(self, pred_masks, target_masks, primitive_length):
        """
        changed by efck: Vectorized IoU matrix computation for all pred-target pairs at once.
        This replaces the O(n^2) Python loop with batched tensor operations.
        
        Args:
            pred_masks: (num_preds, num_primitives) boolean tensor
            target_masks: (num_targets, num_primitives) boolean tensor  
            primitive_length: (num_primitives,) float tensor
            
        Returns:
            iou_matrix: (num_targets, num_preds) float tensor
        """
        # Convert to float for weighted computation
        pred_masks_f = pred_masks.float()  # (num_preds, num_primitives)
        target_masks_f = target_masks.float()  # (num_targets, num_primitives)
        
        # Weight by primitive length
        weighted_pred = pred_masks_f * primitive_length.unsqueeze(0)  # (num_preds, num_primitives)
        weighted_target = target_masks_f * primitive_length.unsqueeze(0)  # (num_targets, num_primitives)
        
        # Compute intersection: (num_targets, num_preds)
        # For each target-pred pair, intersection = sum of lengths where both are 1
        intersection = torch.mm(target_masks_f, (pred_masks_f * primitive_length.unsqueeze(0)).T)
        
        # Compute union: sum_target + sum_pred - intersection
        sum_pred = weighted_pred.sum(dim=1)  # (num_preds,)
        sum_target = weighted_target.sum(dim=1)  # (num_targets,)
        
        # Union = sum_target[:, None] + sum_pred[None, :] - intersection
        union = sum_target.unsqueeze(1) + sum_pred.unsqueeze(0) - intersection
        
        # IoU
        iou_matrix = intersection / (union + torch.finfo(union.dtype).eps)
        
        return iou_matrix

    def eval_semantic_quality(
        self, list_pred_sem_labels, list_target_sem_labels, list_primitive_lens
    ):
        """
        Calculate semantic symbol spotting metrics: F1, wF1

        Args:
            `list_pred_sem_labels` (`List[torch.Tensor]`):
                a list of tensors, each tensor shape is (num_primitives,)
                each value represents the predicted class of primitive
            `list_target_sem_labels` (`List[torch.Tensor]`):
                a list of tensors, each tensor shape is (num_primitives,)
                each value represents the ground truth class of primitive
            `list_primitive_lens` (`List[torch.Tensor]`):
                a list of tensors, each tensor shape is (num_primitives,)
                each value represents the length of primitive
        Returns:
            `Dict[str, float]`: Dictionary containing semantic symbol spotting metrics
                `tp_per_class` (`torch.Tensor`, shape is (num_classes + 1,)): Number of true positive instances for each class
                `pred_per_class` (`torch.Tensor`, shape is (num_classes + 1,)): Number of predicted instances for each class
                `gt_per_class` (`torch.Tensor`, shape is (num_classes + 1,)): Number of ground truth instances for each class
                `w_tp_per_class` (`torch.Tensor`, shape is (num_classes + 1,)): Number of true positive instances for each class
                `w_pred_per_class` (`torch.Tensor`, shape is (num_classes + 1,)): Number of predicted instances for each class
                `w_gt_per_class` (`torch.Tensor`, shape is (num_classes + 1,)): Number of ground truth instances for each class
        """
        tp_per_class = torch.zeros(
            self.num_classes + 1,
            dtype=torch.int32,
            device=list_pred_sem_labels[0].device,
        )
        pred_per_class = torch.zeros(
            self.num_classes + 1,
            dtype=torch.int32,
            device=list_pred_sem_labels[0].device,
        )
        gt_per_class = torch.zeros(
            self.num_classes + 1,
            dtype=torch.int32,
            device=list_pred_sem_labels[0].device,
        )

        w_tp_per_class = torch.zeros(
            self.num_classes + 1,
            dtype=torch.float32,
            device=list_pred_sem_labels[0].device,
        )
        w_pred_per_class = torch.zeros(
            self.num_classes + 1,
            dtype=torch.float32,
            device=list_pred_sem_labels[0].device,
        )
        w_gt_per_class = torch.zeros(
            self.num_classes + 1,
            dtype=torch.float32,
            device=list_pred_sem_labels[0].device,
        )

        for pred_sem_labels, target_sem_labels, primitive_lens in zip(
            list_pred_sem_labels, list_target_sem_labels, list_primitive_lens
        ):
            # changed by efck: vectorized computation using scatter_add instead of Python loop
            # This replaces O(n) Python loop with O(1) GPU scatter operations
            num_classes = self.num_classes + 1
            
            # Count predictions per class using bincount
            pred_counts = torch.bincount(pred_sem_labels.long(), minlength=num_classes)
            gt_counts = torch.bincount(target_sem_labels.long(), minlength=num_classes)
            
            # Weighted counts using scatter_add
            w_pred_counts = torch.zeros(num_classes, dtype=torch.float32, device=pred_sem_labels.device)
            w_gt_counts = torch.zeros(num_classes, dtype=torch.float32, device=pred_sem_labels.device)
            w_pred_counts.scatter_add_(0, pred_sem_labels.long(), primitive_lens.float())
            w_gt_counts.scatter_add_(0, target_sem_labels.long(), primitive_lens.float())
            
            # True positives: where prediction equals target
            correct_mask = pred_sem_labels == target_sem_labels
            correct_labels = pred_sem_labels[correct_mask]
            correct_lengths = primitive_lens[correct_mask]
            
            tp_counts = torch.bincount(correct_labels.long(), minlength=num_classes)
            w_tp_counts = torch.zeros(num_classes, dtype=torch.float32, device=pred_sem_labels.device)
            if correct_labels.numel() > 0:
                w_tp_counts.scatter_add_(0, correct_labels.long(), correct_lengths.float())
            
            # Accumulate
            pred_per_class += pred_counts[:num_classes].to(torch.int32)
            gt_per_class += gt_counts[:num_classes].to(torch.int32)
            tp_per_class += tp_counts[:num_classes].to(torch.int32)
            w_pred_per_class += w_pred_counts[:num_classes]
            w_gt_per_class += w_gt_counts[:num_classes]
            w_tp_per_class += w_tp_counts[:num_classes]

        return dict(
            tp_per_class=tp_per_class,
            pred_per_class=pred_per_class,
            gt_per_class=gt_per_class,
            w_tp_per_class=w_tp_per_class,
            w_pred_per_class=w_pred_per_class,
            w_gt_per_class=w_gt_per_class,
        )

    def eval_instance_quality(self, preds, data_paths):
        """
        `preds` (`Dict[str, List[torch.Tensor]]`): In panoptic quality, the preds should have the following keys:

                \\- `pred_masks` (`List[torch.Tensor]`):
                    a list of tensors, each tensor shape is (num_predictions, num_primitives)
                    each line represents a predicted instance, each column represents a primitive,
                    each value is 0 or 1, 1 means the primitive is part of the instance

                \\- `pred_labels` (`List[torch.Tensor]`):
                    a list of tensors, each tensor shape is (num_predictions,)
                    each value represents the class of the predicted instance
        """
        for batch_idx in range(len(preds["pred_masks"])):
            pred_labels = preds["pred_labels"][batch_idx]
            pred_masks = preds["pred_masks"][batch_idx]
            data_path = data_paths[batch_idx]
            data_split, data_name = data_path.split("/")[-2:]
            output_path = os.path.join(self.output_dir, data_split, data_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            """
            output json format:
            {
                "pred_instances": [
                    {
                       "primitive_ids": [1, 2, 3, ...],
                       "label": 1, 
                       "score": 0.8,
                    },
                    ...
                ]
            }
            """
            output = {"pred_instances": []}
            # changed by efck: batch the .item() calls and .cpu() conversion to reduce GPU sync points
            # Previously each .item() caused a GPU sync, now we transfer all data at once
            pred_labels_cpu = pred_labels.cpu().tolist()
            pred_masks_cpu = pred_masks.cpu()
            for idx in range(len(pred_masks_cpu)):
                pred_label = pred_labels_cpu[idx]
                pred_mask = pred_masks_cpu[idx]
                # get indices where pred_mask is 1
                primitive_ids = pred_mask.nonzero(as_tuple=True)[0].tolist()
                output["pred_instances"].append(
                    {"primitive_ids": primitive_ids, "label": pred_label, "score": 1.0}
                )
            with open(output_path, "w") as f:
                json.dump(output, f)

    def _calculate_primitive_iou(self, pred_mask, target_mask, primitive_length):
        """
        Calculate the IoU between the predicted instance and the ground truth instance

        Args:

            `pred_mask` (`torch.Tensor`, shape is (num_primitives,)):
                each value is 0 or 1, 1 means the primitive is part of the instance

            `target_mask` (`torch.Tensor`, shape is (num_primitives,)):
                each value is 0 or 1, 1 means the primitive is part of the instance

            `primitive_length` (`torch.Tensor`, shape is (num_primitives,)):
                each value represents the length of the primitive

        Returns:

            `torch.Tensor`: IoU between the predicted instance and the ground truth instance
        """
        inter_area = torch.sum(
            primitive_length[torch.logical_and(pred_mask, target_mask)]
        )
        union_area = torch.sum(
            primitive_length[torch.logical_or(pred_mask, target_mask)]
        )
        iou = inter_area / (union_area + torch.finfo(union_area.dtype).eps)
        return iou


@dataclass
class MetricsComputerConfig:
    num_classes: int
    thing_class_idxs: List[int]
    stuff_class_idxs: List[int]
    ignore_label: Optional[List[int]] = None
    log_per_class_metrics: bool = False


class MetricsComputer:
    def __init__(self, config: MetricsComputerConfig) -> None:
        self.num_classes = config.num_classes
        self.thing_class_idxs = config.thing_class_idxs
        self.stuff_class_idxs = config.stuff_class_idxs
        self.ignore_label = config.ignore_label if config.ignore_label is not None else []
        self.log_per_class_metrics = config.log_per_class_metrics
        self.dict_sublosses = {}
        self.metric_states = {}
        self.f1_states = {}

    def __call__(
        self, eval_pred: EvalPrediction, compute_result: bool = False
    ) -> Mapping[str, float]:
        outputs, labels = eval_pred
        dict_sublosses, metric_states, f1_states = outputs

        self._update_dict_sublosses(dict_sublosses)
        self._update_metric_states(metric_states)
        self._update_f1_states(f1_states)
        if not compute_result:
            return
        metrics = {}
        metrics.update(self._get_dict_sublosses())
        metrics.update(self._compute_panoptic_quality())
        metrics.update(self._compute_f1_scores())
        return metrics

    def _update_f1_states(self, f1_states: Dict[str, torch.Tensor]) -> None:
        for key, value in f1_states.items():
            if key not in self.f1_states:
                self.f1_states[key] = (
                    torch.tensor(value)
                    .reshape(-1, len(self.thing_class_idxs + self.stuff_class_idxs) + 1)
                    .sum(dim=0)
                )
            else:
                self.f1_states[key] += (
                    torch.tensor(value)
                    .reshape(-1, len(self.thing_class_idxs + self.stuff_class_idxs) + 1)
                    .sum(dim=0)
                )

    def _compute_f1_scores(self) -> Dict[str, float]:
        results = {}
        eps = torch.finfo(torch.float32).eps

        # Build valid class mask (exclude ignored classes) for F1 computation
        # f1_states tensors have shape (num_classes + 1,) where last element is background
        num_f1_classes = self.num_classes + 1
        valid_f1_mask = torch.ones(num_f1_classes, dtype=torch.bool)
        for il in self.ignore_label:
            if il < num_f1_classes:
                valid_f1_mask[il] = False

        # calculate total F1 and wF1 (excluding ignored classes)
        tp = self.f1_states["tp_per_class"][valid_f1_mask].sum()
        pred = self.f1_states["pred_per_class"][valid_f1_mask].sum()
        gt = self.f1_states["gt_per_class"][valid_f1_mask].sum()
        precision = tp / (pred + eps)
        recall = tp / (gt + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        w_tp = self.f1_states["w_tp_per_class"][valid_f1_mask].sum()
        w_pred = self.f1_states["w_pred_per_class"][valid_f1_mask].sum()
        w_gt = self.f1_states["w_gt_per_class"][valid_f1_mask].sum()
        w_precision = w_tp / (w_pred + eps)
        w_recall = w_tp / (w_gt + eps)
        w_f1 = 2 * w_precision * w_recall / (w_precision + w_recall + eps)

        # changed by efck: batch both .item() calls into single CPU transfer to minimize GPU sync points
        f1_metrics = torch.stack([f1, w_f1]).cpu().tolist()
        results["F1"] = f1_metrics[0]
        results["wF1"] = f1_metrics[1]

        # calculate each class F1 and wF1
        if self.log_per_class_metrics:
            # changed by efck: compute all per-class metrics on GPU first, then batch convert to Python
            # This avoids multiple GPU sync points from individual .item() calls
            f1_per_class = []
            wf1_per_class = []
            for i in range(self.num_classes):
                precision_i = self.f1_states["tp_per_class"][i] / (
                    self.f1_states["pred_per_class"][i] + eps
                )
                recall_i = self.f1_states["tp_per_class"][i] / (
                    self.f1_states["gt_per_class"][i] + eps
                )
                f1_i = 2 * precision_i * recall_i / (precision_i + recall_i + eps)
                f1_per_class.append(f1_i)
                w_precision_i = self.f1_states["w_tp_per_class"][i] / (
                    self.f1_states["w_pred_per_class"][i] + eps
                )
                w_recall_i = self.f1_states["w_tp_per_class"][i] / (
                    self.f1_states["w_gt_per_class"][i] + eps
                )
                w_f1_i = (
                    2 * w_precision_i * w_recall_i / (w_precision_i + w_recall_i + eps)
                )
                wf1_per_class.append(w_f1_i)
            
            # Batch convert all per-class metrics at once
            f1_values = torch.stack(f1_per_class).cpu().tolist()
            wf1_values = torch.stack(wf1_per_class).cpu().tolist()
            for i in range(self.num_classes):
                if i in self.ignore_label:
                    continue
                results[f"class_{i + 1}_F1"] = f1_values[i]
                results[f"class_{i + 1}_wF1"] = wf1_values[i]
            
            precision_bg = self.f1_states["tp_per_class"][-1] / (
                self.f1_states["pred_per_class"][-1] + eps
            )
            recall_bg = self.f1_states["tp_per_class"][-1] / (
                self.f1_states["gt_per_class"][-1] + eps
            )
            f1_bg = 2 * precision_bg * recall_bg / (precision_bg + recall_bg + eps)
            w_precision_bg = self.f1_states["w_tp_per_class"][-1] / (
                self.f1_states["w_pred_per_class"][-1] + eps
            )
            w_recall_bg = self.f1_states["w_tp_per_class"][-1] / (
                self.f1_states["w_gt_per_class"][-1] + eps
            )
            w_f1_bg = (
                2 * w_precision_bg * w_recall_bg / (w_precision_bg + w_recall_bg + eps)
            )
            # changed by efck: batch convert background metrics to reduce GPU sync points
            bg_metrics = torch.stack([f1_bg, w_f1_bg]).cpu().tolist()
            results["class_bg_F1"] = bg_metrics[0]
            results["class_bg_wF1"] = bg_metrics[1]

        self.f1_states.clear()
        return results

    def _update_dict_sublosses(self, dict_sublosses: Dict[str, float]) -> None:
        for key, value in dict_sublosses.items():
            if key not in self.dict_sublosses:
                self.dict_sublosses[key] = {
                    "count": len(value),
                    "sum": value.sum().item(),
                }
            else:
                self.dict_sublosses[key]["count"] += len(value)
                self.dict_sublosses[key]["sum"] += value.sum().item()

    def _get_dict_sublosses(self) -> Dict[str, float]:
        dict_sublosses = {}
        for key, value in self.dict_sublosses.items():
            dict_sublosses[key] = value["sum"] / value["count"]
        self.dict_sublosses.clear()
        return dict_sublosses

    def _update_metric_states(self, metric_states: Dict[str, torch.Tensor]) -> None:
        for key, value in metric_states.items():
            if key not in self.metric_states:
                self.metric_states[key] = (
                    torch.tensor(value)
                    .reshape(-1, len(self.thing_class_idxs + self.stuff_class_idxs))
                    .sum(dim=0)
                )
            else:
                self.metric_states[key] += (
                    torch.tensor(value)
                    .reshape(-1, len(self.thing_class_idxs + self.stuff_class_idxs))
                    .sum(dim=0)
                )

    def _compute_panoptic_quality(self):
        """
        Compute panoptic quality metrics: PQ, SQ, RQ

        Args:

            `metric_states` (`Dict[str, torch.Tensor]`): Dictionary containing intermediate states for metric calculation

                \\- `tp_per_class` (`torch.Tensor`, shape is (num_classes,)): Number of true positive instances for each class

                \\- `fp_per_class` (`torch.Tensor`, shape is (num_classes,)): Number of false positive instances for each class

                \\- `fn_per_class` (`torch.Tensor`, shape is (num_classes,)): Number of false negative instances for each class

                \\- `tp_iou_score_per_class` (`torch.Tensor`, shape is (num_classes,)): Summary of True positive IoU score for each class

            `thing_class_idxs` (`List[int]`): List of thing class indices

            `stuff_class_idxs` (`List[int]`): List of stuff class indices

        Returns:

            `Dict[str, float]`: Dictionary containing panoptic quality metrics
                `PQ` (`float`): Panoptic quality
                `SQ` (`float`): Semantic quality
                `RQ` (`float`): Instance quality
                `thing_PQ` (`float`): Thing panoptic quality
                `thing_SQ` (`float`): Thing semantic quality
                `thing_RQ` (`float`): Thing instance quality
                `stuff_PQ` (`float`): Stuff panoptic quality
                `stuff_SQ` (`float`): Stuff semantic quality
                `stuff_RQ` (`float`): Stuff instance quality
                `class_{id}_PQ` (`float`): Panoptic quality for class `id`
        """
        metric_states = self.metric_states
        thing_class_idxs = self.thing_class_idxs
        stuff_class_idxs = self.stuff_class_idxs
        # Filter out ignored class indices from thing/stuff lists
        valid_thing_idxs = [i for i in thing_class_idxs if i not in self.ignore_label]
        valid_stuff_idxs = [i for i in stuff_class_idxs if i not in self.ignore_label]
        valid_all_idxs = valid_thing_idxs + valid_stuff_idxs
        tp_per_class = metric_states["tp_per_class"].to(torch.float32)
        fp_per_class = metric_states["fp_per_class"].to(torch.float32)
        fn_per_class = metric_states["fn_per_class"].to(torch.float32)
        tp_iou_score_per_class = metric_states["tp_iou_score_per_class"].to(
            torch.float32
        )
        eps = torch.finfo(torch.float32).eps

        def cal_scores(tp, fp, fn, tp_iou_score):
            rq = tp / (tp + 0.5 * fn + 0.5 * fp + eps)
            sq = tp_iou_score / (tp + eps)
            pq = rq * sq
            return pq, sq, rq

        pq_per_class, sq_per_class, rq_per_class = cal_scores(
            tp_per_class, fp_per_class, fn_per_class, tp_iou_score_per_class
        )

        class_metrics = {}
        if self.log_per_class_metrics:
            # changed by efck: batch convert all per-class PQ metrics at once to reduce GPU sync points
            pq_values = (pq_per_class * 100).cpu().tolist()
            class_metrics = {
                f"class_{id + 1}_PQ": pq_values[id]
                for id in range(len(pq_per_class))
                if id not in self.ignore_label
            }

        thing_pq, thing_sq, thing_rq = cal_scores(
            tp_per_class[valid_thing_idxs].sum(),
            fp_per_class[valid_thing_idxs].sum(),
            fn_per_class[valid_thing_idxs].sum(),
            tp_iou_score_per_class[valid_thing_idxs].sum(),
        )

        stuff_pq, stuff_sq, stuff_rq = cal_scores(
            tp_per_class[valid_stuff_idxs].sum(),
            fp_per_class[valid_stuff_idxs].sum(),
            fn_per_class[valid_stuff_idxs].sum(),
            tp_iou_score_per_class[valid_stuff_idxs].sum(),
        )

        pq, sq, rq = cal_scores(
            tp_per_class[valid_all_idxs].sum(),
            fp_per_class[valid_all_idxs].sum(),
            fn_per_class[valid_all_idxs].sum(),
            tp_iou_score_per_class[valid_all_idxs].sum(),
        )

        self.metric_states.clear()

        # changed by efck: batch all .item() calls together to minimize GPU sync points
        # Compute all metrics on GPU, then transfer to CPU once
        all_metrics = torch.stack([pq, sq, rq, thing_pq, thing_sq, thing_rq, stuff_pq, stuff_sq, stuff_rq])
        all_metrics_cpu = (all_metrics * 100).cpu().tolist()
        
        metrics = {
            "PQ": all_metrics_cpu[0],
            "SQ": all_metrics_cpu[1],
            "RQ": all_metrics_cpu[2],
            "thing_PQ": all_metrics_cpu[3],
            "thing_SQ": all_metrics_cpu[4],
            "thing_RQ": all_metrics_cpu[5],
            "stuff_PQ": all_metrics_cpu[6],
            "stuff_SQ": all_metrics_cpu[7],
            "stuff_RQ": all_metrics_cpu[8],
        }
        metrics.update(class_metrics)

        return metrics
