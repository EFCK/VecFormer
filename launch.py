import json
import logging
import warnings
import os

# ------------ apply patches at very beginning ----------- #
from utils import apply_patches

apply_patches()

# -------------- import installed modules ------------- #
from transformers import TrainingArguments
from transformers.utils.logging import (
    set_verbosity_info,
    enable_default_handler,
    enable_explicit_format,
)
import torch.distributed as dist
import wandb

# ----------------- import custom modules ---------------- #
from model import build_model
from data import build_dataset
from utils import get_args

# --------------------- setup logging -------------------- #
enable_default_handler()
enable_explicit_format()

warnings.filterwarnings("ignore")
logger = logging.getLogger("transformers")


def is_main_process():
    """Check if this is the main process (rank 0)"""
    return not dist.is_initialized() or dist.get_rank() == 0


def init_wandb(training_args, model, dataset_splits):
    """Initialize wandb with comprehensive config"""
    if not is_main_process():
        return

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Get model config
    model_config = model.config.to_dict() if hasattr(model.config, "to_dict") else {}

    wandb.init(
        project="VecFormer",
        name=training_args.run_name,
        config={
            # Training
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "eval_batch_size": training_args.per_device_eval_batch_size,
            "learning_rate": training_args.learning_rate,
            "weight_decay": training_args.weight_decay,
            "optimizer": training_args.optim,
            "lr_scheduler_type": str(training_args.lr_scheduler_type),
            "warmup_ratio": training_args.warmup_ratio,
            "fp16": training_args.fp16,
            "bf16": training_args.bf16,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "max_grad_norm": training_args.max_grad_norm,
            "seed": training_args.seed,
            # Model
            "model_type": model_config.get("model_type", "vecformer"),
            "sample_mode": model_config.get("sample_mode", "line"),
            "num_instance_classes": model_config.get("num_instance_classes", 35),
            "num_semantic_classes": model_config.get("num_semantic_classes", 35),
            "use_layer_fusion": model_config.get("use_layer_fusion", True),
            "backbone_in_channels": model_config.get("backbone_config", {}).get(
                "in_channels", 7
            ),
            "cad_decoder_embed_dim": model_config.get("cad_decoder_config", {}).get(
                "embed_dim", 256
            ),
            "cad_decoder_n_blocks": model_config.get("cad_decoder_config", {}).get(
                "n_blocks", 6
            ),
            "cad_decoder_n_heads": model_config.get("cad_decoder_config", {}).get(
                "n_heads", 8
            ),
            # Parameters
            "total_params_M": total_params / 1e6,
            "trainable_params_M": trainable_params / 1e6,
            # Data
            "train_samples": len(dataset_splits.train),
            "val_samples": len(dataset_splits.val),
            "test_samples": len(dataset_splits.test),
            # Loss weights
            "class_loss_weight": model_config.get("instance_criterion_config", {}).get(
                "class_loss_weight", 2.5
            ),
            "bce_loss_weight": model_config.get("instance_criterion_config", {}).get(
                "bce_loss_weight", 5.0
            ),
            "dice_loss_weight": model_config.get("instance_criterion_config", {}).get(
                "dice_loss_weight", 5.0
            ),
            "ce_loss_weight": model_config.get("semantic_criterion_config", {}).get(
                "ce_loss_weight", 5.0
            ),
            # Evaluation
            "eval_strategy": str(training_args.eval_strategy),
            "eval_steps": training_args.eval_steps,
            "metric_for_best_model": training_args.metric_for_best_model,
            # Environment
            "launch_mode": training_args.launch_mode,
            "output_dir": training_args.output_dir,
        },
        reinit=True,
    )

    logger.info("Wandb initialized successfully")


def finish_wandb():
    """Finish wandb run"""
    if is_main_process():
        wandb.finish()


# ----------------------- main func ---------------------- #
def main():
    # --------------------- parse args -------------------- #
    training_args = get_args(TrainingArguments)[0]
    # ------------------- setup logging ------------------ #
    if training_args.should_log:
        # The default of training_args.log_level is passive,
        # so we set log level at info here to have that default.
        set_verbosity_info()

    logger.info(f"Training Arguments: {json.dumps(training_args.to_dict(), indent=4)}")
    # ----------------------- model ---------------------- #
    model, ModelTrainer = build_model(training_args.model_args_path)
    # ---------------------- dataset --------------------- #
    dataset_splits, data_collator = build_dataset(training_args.data_args_path)
    # ---------------------- trainer --------------------- #
    trainer = ModelTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_splits.train,
        eval_dataset=dataset_splits.test
        if training_args.launch_mode == "test"
        else dataset_splits.val,
        data_collator=data_collator,
    )  # type: ignore
    # -------------------- init wandb -------------------- #
    init_wandb(training_args, model, dataset_splits)
    # ----------------------- train ---------------------- #
    if training_args.launch_mode == "train":
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
            logger.info(f"Resuming from checkpoint: {checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.log_metrics("train", train_result.metrics)
    # ------------------ continue train ------------------ #
    if training_args.launch_mode == "continue":
        if training_args.resume_from_checkpoint is None:
            raise ValueError("resume_from_checkpoint is required for continue mode")
        logger.info(
            f"Continuing from checkpoint: {training_args.resume_from_checkpoint}"
        )
        trainer._load_from_checkpoint(training_args.resume_from_checkpoint)
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
    # ----------------------- test ----------------------- #
    if training_args.launch_mode == "test":
        if training_args.resume_from_checkpoint is None:
            raise ValueError("resume_from_checkpoint is required for test mode")
        logger.info(f"Testing from checkpoint: {training_args.resume_from_checkpoint}")
        trainer._load_from_checkpoint(training_args.resume_from_checkpoint)
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
    # ---------------------------------------------------- #
    if training_args.launch_mode not in ["train", "continue", "test"]:
        raise ValueError(f"Invalid launch mode: {training_args.launch_mode}")
    # ------------------- finish wandb ------------------- #
    finish_wandb()


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    main()
    dist.destroy_process_group()
