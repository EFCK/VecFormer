from typing import Dict, Optional

import torch
import torch.distributed as dist
from transformers import Trainer, TrainerCallback

from .configuration_vecformer import VecFormerConfig
from .evaluator import MetricsComputer, MetricsComputerConfig

# Check if wandb is available
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def is_main_process():
    """Check if this is the main process (rank 0)"""
    return not dist.is_initialized() or dist.get_rank() == 0


class LogHistoryCleanupCallback(TrainerCallback):
    """
    Callback to prevent trainer_state.json from bloating.

    HuggingFace Trainer accumulates all logs in state.log_history, which gets
    saved to trainer_state.json on every checkpoint. For long training runs
    (e.g., 100 epochs with logging every 50 steps), this can grow to tens of
    thousands of entries, causing memory issues and slow checkpoint saves.

    This callback keeps only evaluation logs in the history, removing training
    step logs which are already sent to wandb/tensorboard.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Clean up log_history after each log event"""
        if not state.is_world_process_zero:
            return

        # Keep only evaluation logs (those with 'eval_' keys) in log_history
        # Training step logs are still visible in wandb/tensorboard, just not
        # saved to trainer_state.json
        if state.log_history:
            eval_logs = [
                log
                for log in state.log_history
                if any(key.startswith("eval_") for key in log.keys())
            ]
            state.log_history.clear()
            state.log_history.extend(eval_logs)


class WandbBestMetricCallback(TrainerCallback):
    """Callback to track and log best metrics to wandb"""

    def __init__(self, metric_for_best_model: str = "PQ"):
        self.metric_for_best_model = metric_for_best_model
        self.best_metric = 0.0
        self.best_epoch = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation"""
        if not WANDB_AVAILABLE or not is_main_process() or metrics is None:
            return

        # Get the metric value (handle both 'eval_PQ' and 'PQ' formats)
        metric_key = f"eval_{self.metric_for_best_model}"
        metric_value = metrics.get(
            metric_key, metrics.get(self.metric_for_best_model, 0.0)
        )

        current_epoch = state.epoch if state.epoch else 0

        # Update best metric if improved
        if metric_value > self.best_metric:
            self.best_metric = metric_value
            self.best_epoch = current_epoch
            # Update wandb summary
            wandb.run.summary["best_" + self.metric_for_best_model] = self.best_metric
            wandb.run.summary["best_epoch"] = self.best_epoch

        # Log current best metric for chart visualization
        wandb.log(
            {
                f"eval/best_{self.metric_for_best_model}": self.best_metric,
                "epoch": current_epoch,
            },
            step=state.global_step,
        )


class VecFormerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            compute_metrics=MetricsComputer(
                MetricsComputerConfig(**VecFormerConfig().metrics_computer_config)
            ),
            **kwargs,
        )
        self.label_names = [
            "sem_ids",
            "inst_ids",
            "prim_lengths",
            "cu_numprims",
            "data_paths",
        ]
        self.custom_logs: Dict[str, torch.Tensor] = {}
        self.custom_logs_accumulated_step: Dict[str, int] = {}
        self.custom_logs_is_training: bool = False

        # Add wandb best metric callback if wandb is available
        if WANDB_AVAILABLE:
            metric_for_best = (
                self.args.metric_for_best_model
                if self.args.metric_for_best_model
                else "PQ"
            )
            self.add_callback(
                WandbBestMetricCallback(metric_for_best_model=metric_for_best)
            )

        # Add callback to prevent trainer_state.json from bloating
        # This removes training step logs from state.log_history, keeping only eval logs
        self.add_callback(LogHistoryCleanupCallback())

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        # ----------- hack to log multiple loss ---------- #
        if self.custom_logs_is_training:
            if self.custom_logs:
                stacked_values = torch.stack(list(self.custom_logs.values()))
                gathered_values = self._nested_gather(stacked_values).view(
                    dist.get_world_size(), -1
                )
                mean_values = gathered_values.mean(dim=0)
                for key, value in zip(self.custom_logs.keys(), mean_values):
                    logs[key] = round(
                        value.item() / (self.custom_logs_accumulated_step[key]), 4
                    )
                self.custom_logs.clear()
        # ------------------------------------------------ #
        super().log(logs, start_time)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        # ----------- hack to log multiple loss ---------- #
        is_training = (
            return_outputs is False
        )  # in huggingface trainer, return_outputs is True only when evaluating
        self.custom_logs_is_training = is_training is True
        if is_training:
            dict_sublosses = outputs["dict_sublosses"]
            for key, value in dict_sublosses.items():
                if key in self.custom_logs:
                    self.custom_logs[key] += value
                    self.custom_logs_accumulated_step[key] += 1
                else:
                    self.custom_logs[key] = value
                    self.custom_logs_accumulated_step[key] = 1
        # ------------------------------------------------ #
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # hack to handle `nested_detach` stuck in huggingface trainer
        losses, logits, _ = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )
        # Handle case where losses could be None
        if losses is not None:
            device = losses.device
        elif logits is not None:
            # Try to get device from logits
            if isinstance(logits, torch.Tensor):
                device = logits.device
            elif isinstance(logits, (tuple, list)) and len(logits) > 0:
                device = (
                    logits[0].device if isinstance(logits[0], torch.Tensor) else "cuda"
                )
            else:
                device = "cuda"
        else:
            device = "cuda"
        return (
            losses,
            logits,
            torch.tensor(0.0, dtype=torch.float32, device=device),
        )
