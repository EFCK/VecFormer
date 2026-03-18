import logging
from typing import Dict, Optional

import torch
import torch.distributed as dist
from transformers import Trainer, TrainerCallback

logger = logging.getLogger(__name__)

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


# changed by efck - 2026-03-18 - aim: fix LR being 1e-6 instead of config 1e-5 on resume;
# root cause: _load_optimizer_and_scheduler ran before state.max_steps was set (max_steps=-1),
# making warmup_stable_decay decay formula malformed and clamping LR to min_lr_ratio.
# fix: defer scheduler recreation to on_train_begin where state.max_steps is always correct.
class ResumeSchedulerFixCallback(TrainerCallback):
    """
    Deferred scheduler fix for resume runs.

    _load_optimizer_and_scheduler is called before HF Trainer reliably sets
    state.max_steps (in some versions the checkpoint's trainer_state.json
    replaces the state object, leaving max_steps=-1 until the training loop
    computes it). This callback fires in on_train_begin, where state.max_steps
    is always correctly set, and recreates the scheduler with the right total
    steps + the right starting position.
    """

    def __init__(self, trainer: "VecFormerTrainer"):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        if not getattr(self.trainer, "_needs_scheduler_fix", False):
            return
        if state.max_steps <= 0:
            logger.warning(
                "ResumeSchedulerFixCallback: state.max_steps not yet set, skipping scheduler fix."
            )
            return

        import warnings

        config_lr = args.learning_rate
        checkpoint_step = self.trainer._resume_checkpoint_step

        # Recreate the scheduler with the correct total step count.
        # Restoring last_epoch to checkpoint_step places the scheduler in the
        # stable phase (checkpoint_step is between warmup and decay for typical
        # schedules), giving LR = config_lr immediately.
        # Decay starts at total_steps - num_decay_steps from the beginning of
        # the schedule, which equals remaining_steps - num_decay_steps from now.
        self.trainer.lr_scheduler = None
        self.trainer.create_scheduler(
            num_training_steps=state.max_steps, optimizer=self.trainer.optimizer
        )

        # Advance to checkpoint_step: set last_epoch = step - 1, then step() → step
        if hasattr(self.trainer.lr_scheduler, "last_epoch"):
            self.trainer.lr_scheduler.last_epoch = max(0, checkpoint_step - 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.trainer.lr_scheduler.step()  # → last_epoch = checkpoint_step, LR = config_lr

        # Keep callback_handler's reference in sync with the new scheduler object
        if hasattr(self.trainer, "callback_handler"):
            self.trainer.callback_handler.lr_scheduler = self.trainer.lr_scheduler

        self.trainer._needs_scheduler_fix = False

        num_decay_steps = (args.lr_scheduler_kwargs or {}).get("num_decay_steps", 0)
        decay_in = state.max_steps - checkpoint_step - num_decay_steps
        logger.warning(
            f"Recreated scheduler: LR={config_lr}, max_steps={state.max_steps}, "
            f"restored to step={checkpoint_step}. Decay starts in {decay_in} steps."
        )


class VecFormerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Use the actual model's config for MetricsComputer so that ignore_label
        # and other settings match what the model was built with.
        # Fall back to VecFormerConfig() defaults if model is not provided.
        # made by EFCK: previously always used VecFormerConfig() (defaults), which
        # meant ignore_label was always [35] regardless of the model config YAML.
        model = kwargs.get("model") or (args[0] if args else None)
        mc_config = (
            model.config.metrics_computer_config
            if model is not None and hasattr(model, "config")
            else VecFormerConfig().metrics_computer_config
        )
        super().__init__(
            *args,
            compute_metrics=MetricsComputer(MetricsComputerConfig(**mc_config)),
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

        # Flags used by _load_optimizer_and_scheduler + ResumeSchedulerFixCallback
        self._needs_scheduler_fix: bool = False
        self._resume_checkpoint_step: int = 0

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

        # Deferred scheduler fix for resume runs (no-op on fresh training)
        self.add_callback(ResumeSchedulerFixCallback(trainer=self))

    # changed by efck - 2026-03-18 - aim: override HF trainer resume to allow LR and scheduler updates
    def _load_optimizer_and_scheduler(self, checkpoint):
        """
        Override HuggingFace Trainer's optimizer and scheduler loading to allow
        updating learning rate and decay steps upon resuming.

        Only the optimizer LRs are updated here. Scheduler recreation is deferred
        to ResumeSchedulerFixCallback.on_train_begin where state.max_steps is
        reliably set (avoids the max_steps=-1 bug from earlier approach).
        """
        super()._load_optimizer_and_scheduler(checkpoint)

        if self.optimizer is None or self.lr_scheduler is None:
            return

        config_lr = self.args.learning_rate

        # Update optimizer base learning rates
        for group in self.optimizer.param_groups:
            if "initial_lr" in group:
                group["initial_lr"] = config_lr
            group["lr"] = config_lr

        # Save checkpoint step for the deferred scheduler fix
        self._resume_checkpoint_step = self.lr_scheduler.last_epoch
        self._needs_scheduler_fix = True

        logger.warning(
            f"Optimizer LR overridden to {config_lr}. "
            f"Scheduler will be recreated on_train_begin "
            f"(checkpoint_step={self._resume_checkpoint_step})."
        )

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

    def training_step(self, model, inputs, num_items_in_batch=None):
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise
            torch.cuda.empty_cache()
            # Zero out any partial gradients so the optimizer step is a no-op
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            data_paths = inputs.get("data_paths", [])
            logger.warning(
                f"[OOM] Skipped training batch (step {self.state.global_step}). "
                f"Files: {data_paths}"
            )
            # Return a zero loss so the Trainer's gradient scaler / optimizer step
            # proceeds without crashing.  Gradients are already cleared above.
            device = next(model.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=False)

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
