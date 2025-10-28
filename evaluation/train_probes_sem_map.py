import datetime
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate, get_original_cwd
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import (BarColumn, Progress, SpinnerColumn,
                           TaskProgressColumn, TextColumn)
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy, JaccardIndex
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

from utils.training import get_batch, get_dataloaders
from utils.visualization import UnNormalize

LOG_INTERVAL = 100


class SegmentationEvaluator:

    def __init__(self, model, backbone, device, cfg, writer, console):
        self.model, self.backbone, self.device, self.cfg, self.writer, self.console = (
            model,
            backbone,
            device,
            cfg,
            writer,
            console,
        )

        self.mean = backbone.config["mean"]
        self.std = backbone.config["std"]

        # Initialize segmentation-specific components
        self.accuracy_metric = Accuracy(num_classes=cfg.metrics.seg.num_classes, task="multiclass").to(device)
        self.iou_metric = JaccardIndex(num_classes=cfg.metrics.seg.num_classes, task="multiclass").to(device)
        self.classifier = nn.Conv2d(cfg.model.feature_dim, cfg.metrics.seg.num_classes, 1).to(device)

    def set_up_classifier(self, checkpoint_path):
        """Load classifier weights from a checkpoint."""
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            self.classifier.load_state_dict(checkpoint["model_state_dict"])
            self.console.print(f"Loaded classifier from checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    def set_optimizer(self, cfg):
        params = []
        params_classifier = self.classifier.parameters()
        params_model = self.model.parameters()

        params = list(params_classifier)
        optimizer = instantiate(cfg.optimizer, params=params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Log number of parameters
        num_params = sum(p.numel() for p in params if p.requires_grad)
        self.log_print(f"[bold cyan]Number of optimized parameters: {num_params:,}[/bold cyan]")

    def log_print(self, *args, **kwargs):
        """Log to both file and terminal with immediate updates"""
        # Write to terminal
        Console(force_terminal=True).print(*args, **kwargs)
        # Write to file and flush
        self.console.print(*args, **kwargs)
        if hasattr(self.console, "file") and self.console.file:
            self.console.file.flush()

    def log_tensorboard(self, step, loss=None, metrics=None):
        """Log losses and metrics to TensorBoard."""
        if loss is not None:
            self.writer.add_scalar("Loss/Step", loss, step)
        if metrics is not None:
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(f"Metrics/{metric_name}", metric_value, step)

    def process_batch(self, image_batch, target, is_training=True):
        H, W = target.shape[-2:]
        with torch.no_grad():
            pred = self.backbone(image_batch)
            patch_tokens, cls_token = pred[0], pred[1]

            pred = self.model(image_batch, patch_tokens, (H, W))

        pred = self.classifier(pred)

        # Some baselines upsample more than the required target size
        if pred.shape[-2:] != (H, W):
            pred = F.interpolate(pred, size=(H, W), mode="bilinear")

        if target.shape[-2:] != pred.shape[-2:]:
            target = (
                F.interpolate(
                    target.unsqueeze(1),
                    size=pred.shape[-2:],
                    mode="nearest-exact",
                )
                .squeeze(1)
                .to(target.dtype)
            )

        # Create mask for valid pixels (not 255)
        valid_mask = target != 255

        # Reshape predictions and targets
        pred = rearrange(pred, "b c h w -> (b h w) c")
        target = rearrange(target, "b h w -> (b h w)")
        valid_mask = rearrange(valid_mask, "b h w -> (b h w)")

        # Apply mask to both pred and target
        pred = pred[valid_mask]
        target = target[valid_mask]

        return pred, target

    def train(
        self,
        train_dataloader,
        progress,
        epoch,
        start_time,
    ):
        self.log_print(f"[yellow]Training model epoch {epoch+1}...[/yellow]")
        self.backbone.eval()
        self.model.eval()
        self.classifier.train()

        epoch_task = progress.add_task(
            f"Epoch {epoch+1}/{self.cfg.num_epochs}",
            total=len(train_dataloader),
            loss=0.0,
            step=0,
        )
        total_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            # Process batch using get_batch
            batch = get_batch(batch, self.device)
            image_batch = batch["image"]
            target = batch["label"].to(self.device)

            if random.random() < 0.5:
                image_batch = torch.flip(image_batch, dims=[3])  # Flip along width (W)
                target = torch.flip(target, dims=[2])  # Flip along width (W), assuming (H, W) or (1, H, W)

            self.optimizer.zero_grad()

            pred, target = self.process_batch(image_batch, target, is_training=True)

            loss = F.cross_entropy(pred, target)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            avg_loss = total_loss / (batch_idx + 1)

            # Update progress only every log_interval iterations
            if (batch_idx + 1) % LOG_INTERVAL == 0 or batch_idx == len(train_dataloader) - 1:
                elapsed_time = datetime.datetime.now() - start_time
                elapsed_str = str(elapsed_time).split(".")[0]
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Update progress bar
                progress.update(
                    epoch_task,
                    advance=LOG_INTERVAL,
                    loss=avg_loss,
                    step=batch_idx + 1,
                )

                # Log with learning rate
                self.log_print(
                    f"[cyan]Iteration {batch_idx + 1}[/cyan] - "
                    f"Loss: {avg_loss:.6f} - "
                    f"LR: {current_lr:.5e} - "
                    f"Elapsed Time: {elapsed_str}"
                )

                # Force console update
                if self.console and hasattr(self.console, "file"):
                    self.console.file.flush()

                # Ensure progress is displayed immediately
                progress.refresh()

                # Log loss to TensorBoard
                self.log_tensorboard(len(train_dataloader) + batch_idx, loss=avg_loss)

            if self.cfg.sanity and batch_idx == 0:
                break

            self.scheduler.step()

            if self.cfg.sanity:
                break

        # Add learning rate to epoch summary
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.log_print(
            f"[bold cyan]Epoch {epoch+1} Summary:[/bold cyan] " f"Loss = {avg_loss:.6f} - " f"LR = {current_lr:.2e}"
        )

        return

    def save_checkpoint(self, checkpoint_path):
        console = self.console
        # Save final model state after training
        checkpoint = {
            "epoch": self.cfg.num_epochs - 1,
            "model_state_dict": self.classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "task": "seg",
            "backbone": self.cfg.backbone.name,
        }
        torch.save(checkpoint, checkpoint_path)
        self.log_print(f"[bold green]Training completed. Model saved at: {checkpoint_path}[/bold green]")
        console.file.flush()
        return

    @torch.inference_mode()
    def evaluate(self, dataloader, epoch):
        self.log_print("[yellow]Evaluating model...[/yellow]")
        torch.cuda.empty_cache()

        self.backbone.eval()
        self.model.eval()
        self.classifier.eval()

        # Reset metrics at the start of evaluation
        self.accuracy_metric.reset()
        self.iou_metric.reset()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch = get_batch(batch, self.device)
            image_batch = batch["image"]
            target = batch["label"].to(self.device)

            # Process batch and get masked predictions and targets
            pred, target = self.process_batch(image_batch, target, is_training=False)

            self.accuracy_metric(pred, target)
            self.iou_metric(pred, target)

            if self.cfg.sanity and batch_idx == 0:
                break

        metrics = {
            "accuracy": self.accuracy_metric.compute().item(),
            "iou": self.iou_metric.compute().item(),
        }

        # Log metrics to TensorBoard
        self.log_tensorboard(step=epoch, metrics=metrics)

        self.log_print(f"[bold green]Results: {metrics}[/bold green]")
        return

    @torch.inference_mode()
    def simple_inference(self, image_batch):
        self.backbone.eval()
        self.model.eval()
        self.classifier.eval()

        H, W = image_batch.shape[-2:]
        with torch.no_grad():
            hr_feats, _ = self.backbone(image_batch)
            features = self.model(image_batch, hr_feats, (H, W))

        pred = features  # Get the last prediction
        pred = self.classifier(pred)  # Pass through the classifier
        pred = pred.argmax(dim=1)  # Get the predicted class for each pixel

        return pred, features, hr_feats


@hydra.main(config_path="../config", config_name="eval")
def main(cfg):
    task = "seg"  # Force segmentation task

    # Setup Classifier
    # Either we train one classifier per backbone.
    checkpoint_dir = f"./checkpoints/{task}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/{cfg.model.name}.pth"

    # Create persistent consoles instead of creating new ones each time
    terminal_console = Console()  # Terminal output
    if Path(checkpoint_path).exists():
        file_name = f"eval_{cfg.model.name}_{task}.log"
    else:
        file_name = f"train_{cfg.model.name}_{task}.log"
    current_run_dir = os.getcwd()
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    source_path = os.path.join(current_run_dir, file_name)
    symlink_path = os.path.join(current_run_dir, log_dir, file_name)
    if not os.path.exists(symlink_path):
        os.symlink(source_path, symlink_path)
    file_console = Console(file=open(source_path, "w"))

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(current_run_dir, "./tb", file_name.replace(".log", "")))

    def log_print(*args, **kwargs):
        """Log to both terminal and file with immediate flushing"""
        terminal_console.print(*args, **kwargs)
        file_console.print(*args, **kwargs)
        file_console.file.flush()

    # Start logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_print(f"\n[bold blue]{'='*50}[/bold blue]")
    log_print(f"[bold blue]Starting at {timestamp}[/bold blue]")
    log_print(f"[bold green]Configuration:[/bold green]")
    log_print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"[bold yellow]Using device: {device}[/bold yellow]")

    log_print(f"\n[bold cyan]Processing {task} task:[/bold cyan]")
    log_print(f"\n[bold cyan]Image size: {cfg.img_size}[/bold cyan]")

    # Setup Backbone
    backbone = instantiate(cfg.backbone).to(device)
    backbone.requires_grad_(False)
    backbone.eval()

    # Setup Model
    model = instantiate(cfg.model).to(device)
    if cfg.eval.model_ckpt:
        model_ckpt_path = cfg.eval.model_ckpt
        if not os.path.isabs(model_ckpt_path):
            model_ckpt_path = os.path.join(get_original_cwd(), model_ckpt_path.lstrip("/"))
        if not os.path.exists(model_ckpt_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt_path}")
        checkpoint = torch.load(model_ckpt_path, weights_only=False)
        if cfg.model.name == "jafar":
            model.load_state_dict(checkpoint["jafar"], strict=False)
        log_print(f"[green]Loaded model from checkpoint: {model_ckpt_path}[/green]")
    else:
        model.train()

    # Setup Dataloaders
    train_loader, val_loader = get_dataloaders(cfg, backbone, is_evaluation=True, mean=None, std=None)
    log_print(f"[bold cyan]Train Dataset size: {len(train_loader.dataset)}[/bold cyan]")
    log_print(f"[bold cyan]Val Dataset size: {len(val_loader.dataset)}[/bold cyan]")

    # Setup Evaluator
    evaluator = SegmentationEvaluator(model, backbone, device, cfg, writer, file_console)

    # Already trained
    if Path(checkpoint_path).exists():
        log_print(f"[green]Loading classifier from {checkpoint_path}[/green]")
        evaluator.set_up_classifier(checkpoint_path)
        evaluator.evaluate(val_loader, epoch=0)
    else:
        log_print(f"[yellow]Training classifier... {checkpoint_path} not found[/yellow]\n")
        evaluator.set_optimizer(cfg)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[yellow]Loss: {task.fields[loss]:.6f}"),
            TextColumn("[green]Step: {task.fields[step]:.6e}"),
            console=file_console,
        )

        start_time = datetime.datetime.now()

        with progress:
            # Standard training for upsampler
            log_print(f"[yellow]Training for {cfg.num_epochs} epochs[/yellow]\n")

            for epoch in range(cfg.num_epochs):
                evaluator.train(train_loader, progress, epoch, start_time)
                evaluator.evaluate(val_loader, epoch)
                
                # Save checkpoint after each epoch
                epoch_checkpoint_path = checkpoint_path.replace('.pth', f'_epoch_{epoch+1}.pth')
                evaluator.save_checkpoint(epoch_checkpoint_path)
                
                if cfg.sanity:
                    break

            # Save final checkpoint with original name
            evaluator.save_checkpoint(checkpoint_path)

    file_console.file.close()
    writer.close()  # Close TensorBoard writer


if __name__ == "__main__":
    main()
