#!/usr/bin/env python
"""Train the student vision transformer on CIFAR-10.

This script is intentionally separate from the model implementation. Importing
it has no side effects: CIFAR-10 is downloaded and training starts only from
main(), after the user runs this script explicitly.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from qwen25_scratch.vision_steps.config import StudentVisionConfig
from qwen25_scratch.vision_steps.vision_transformer import StudentVisionClassifier


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path
    output_dir: Path
    image_size: int = 56
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    num_workers: int = 4
    seed: int = 7
    download: bool = False
    device: str = "auto"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Return training augmentation and deterministic evaluation preprocessing."""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_transform, evaluation_transform


def build_dataloaders(config: TrainConfig, device: torch.device) -> tuple[
    DataLoader, DataLoader, DataLoader
]:
    """Build a deterministic 45k/5k train/validation split plus the test set."""
    train_transform, evaluation_transform = build_transforms(config.image_size)

    augmented_training_set = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        transform=train_transform,
        download=config.download,
    )
    evaluation_training_set = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        transform=evaluation_transform,
        download=False,
    )
    test_set = datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        transform=evaluation_transform,
        download=config.download,
    )

    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(len(augmented_training_set), generator=generator).tolist()
    train_indices, validation_indices = indices[:45_000], indices[45_000:]

    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(
        Subset(augmented_training_set, train_indices),
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        Subset(evaluation_training_set, validation_indices),
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, validation_loader, test_loader


def run_epoch(
    model: StudentVisionClassifier,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: AdamW | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float]:
    """Run one training or evaluation epoch and return mean loss and accuracy."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    progress = tqdm(batches, desc="train" if training else "evaluate", leave=False)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        use_amp = device.type == "cuda"
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images).logits
            loss = criterion(logits, labels)

        if training:
            if scaler is None:
                raise RuntimeError("training requires a gradient scaler")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_examples += batch_size
        progress.set_postfix(
            loss=f"{total_loss / total_examples:.4f}",
            accuracy=f"{total_correct / total_examples:.3f}",
        )

    return total_loss / total_examples, total_correct / total_examples


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: StudentVisionClassifier,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    model_config: StudentVisionConfig,
    train_config: TrainConfig,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "model_config": asdict(model_config),
            "train_config": {
                **asdict(train_config),
                "data_dir": str(train_config.data_dir),
                "output_dir": str(train_config.output_dir),
            },
            "metrics": metrics,
        },
        path,
    )


def train(config: TrainConfig) -> None:
    """Train, checkpoint each epoch, and evaluate the best checkpoint once."""
    if config.image_size % 14:
        raise ValueError("image_size must be divisible by patch_size=14")

    seed_everything(config.seed)
    device = select_device(config.device)
    print(f"device: {device}")
    print(f"checkpoints: {config.output_dir.resolve()}")

    train_loader, validation_loader, test_loader = build_dataloaders(config, device)
    model_config = StudentVisionConfig(
        image_height=config.image_size,
        image_width=config.image_size,
        patch_size=14,
        hidden_size=64,
        num_heads=4,
        intermediate_size=192,
        depth=2,
    )
    model = StudentVisionClassifier(model_config, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_validation_accuracy = float("-inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
        )
        with torch.inference_mode():
            validation_loss, validation_accuracy = run_epoch(
                model, validation_loader, criterion, device
            )
        scheduler.step()

        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append({"epoch": float(epoch), **metrics})
        print(
            f"epoch {epoch:02d}/{config.epochs}: "
            f"train loss={train_loss:.4f}, accuracy={train_accuracy:.3%}; "
            f"validation loss={validation_loss:.4f}, "
            f"accuracy={validation_accuracy:.3%}"
        )

        save_checkpoint(
            config.output_dir / "latest.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_config=model_config,
            train_config=config,
            metrics=metrics,
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            save_checkpoint(
                config.output_dir / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                model_config=model_config,
                train_config=config,
                metrics=metrics,
            )

        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )

    best_checkpoint = torch.load(
        config.output_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    with torch.inference_mode():
        test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)
    print(f"final test: loss={test_loss:.4f}, accuracy={test_accuracy:.3%}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "cifar10"
    )
    parser.add_argument("--image-size", type=int, default=56)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--download",
        action="store_true",
        help="download CIFAR-10 into --data-dir when it is not already present",
    )
    arguments = parser.parse_args()
    return TrainConfig(
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        image_size=arguments.image_size,
        batch_size=arguments.batch_size,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        num_workers=arguments.num_workers,
        seed=arguments.seed,
        download=arguments.download,
        device=arguments.device,
    )


if __name__ == "__main__":
    train(parse_args())
