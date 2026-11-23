"""Paired CIFAR-10 comparison of two residual updates versus one outer update.

Run --smoke_test for a tiny synthetic check without torchvision or downloads.
"""

import argparse
import csv
import hashlib
import json
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from architectures import load_arch
from architectures.custommlpmixer import MLPMixer as SingleSkipMixer
from architectures.mlpmixer import MLPMixer as OriginalMixer
from methods import load_method
from utils import save_training_curves, set_seed


VARIANTS = {
    "original": "mlpmixerS4",
    "single_skip": "custommlpmixerS4",
}


def parse_args():
    """Read shared experimental settings; both variants receive the same values."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--save_dir", default="./runs/mixer-comparison")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        args.epochs = 2
        args.batch_size = 4
        args.num_workers = 0
        args.seeds = [42]
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch_size must be positive")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must be distinct")
    return args


def seed_worker(worker_id):
    """Seed Python/NumPy augmentation RNGs from each worker's PyTorch seed."""
    import numpy as np

    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_datasets(args):
    """Return train/validation/test datasets and the fixed held-out indices.

    Real runs split CIFAR's training set 45,000/5,000. The official test set is
    separate and is not used for selecting epochs or hyperparameters.
    """
    if args.smoke_test:
        generator = torch.Generator().manual_seed(args.split_seed)
        images = torch.randn(24, 3, 32, 32, generator=generator)
        labels = torch.randint(10, (24,), generator=generator)
        train = TensorDataset(images[:12], labels[:12])
        val = TensorDataset(images[12:18], labels[12:18])
        test = TensorDataset(images[18:], labels[18:])
        return train, val, test, list(range(12, 18))

    from torchvision import datasets, transforms

    stats = ((0.491, 0.482, 0.447), (0.247, 0.244, 0.262))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(*stats),
    ])
    train = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
    val = datasets.CIFAR10(args.data_dir, train=True, download=False, transform=eval_transform)
    test = datasets.CIFAR10(args.data_dir, train=False, download=False, transform=eval_transform)
    indices = torch.randperm(50000, generator=torch.Generator().manual_seed(args.split_seed)).tolist()
    return Subset(train, indices[5000:]), Subset(val, indices[:5000]), test, indices[:5000]


def build_loaders(datasets, args, seed, device):
    """Build fresh loaders for each trial with an independent seeded shuffle RNG."""
    loaders = []
    for index, dataset in enumerate(datasets):
        loaders.append(DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(index == 0),
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed + index),
        ))
    return loaders


@torch.inference_mode()
def evaluate(model, loader, device):
    """Evaluate [B,3,32,32] images against labels; return weighted loss/accuracy."""
    model.eval()
    loss_sum = correct = samples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        samples += labels.size(0)
    return {"loss": loss_sum / samples, "accuracy": correct / samples}


def initial_hash(model):
    """Fingerprint all initial weights so paired runs can verify identical starts."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_trial(variant, seed, args, datasets, device, output_dir):
    """Train one variant/seed, select by validation, and test its selected model once."""
    set_seed(seed)
    if args.smoke_test:
        model_class = OriginalMixer if variant == "original" else SingleSkipMixer
        backbone = model_class(3, 4, 2, 32, 64, 32, 32, 32)
    else:
        backbone, _ = load_arch(VARIANTS[variant])
    model = load_method("supervised", backbone, backbone.feature_dim, 10).to(device)
    fingerprint = initial_hash(model)
    parameter_count = sum(p.numel() for p in model.parameters())
    train_loader, val_loader, test_loader = build_loaders(datasets, args, seed, device)
    # Separate the training RNG from model construction and checkpoint I/O.
    set_seed(seed + 1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)], gamma=0.1,
    )
    trial_dir = output_dir / f"{variant}_seed{seed}"
    trial_dir.mkdir()
    history = []
    best_accuracy = -1.0
    config = dict(vars(args), variant=variant, seed=seed, initial_hash=fingerprint,
                  parameter_count=parameter_count, optimizer="adamw", head="linear")
    (trial_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
              "learning_rate", "train_seconds"]
    with (trial_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            lr = optimizer.param_groups[0]["lr"]
            loss_sum = correct = samples = 0
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss, logits = model.training_step((images, labels), return_logits=True)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * labels.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                samples += labels.size(0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            train_seconds = time.perf_counter() - start
            scheduler.step()
            val_metrics = evaluate(model, val_loader, device)
            row = dict(epoch=epoch, train_loss=loss_sum / samples,
                       train_accuracy=correct / samples, val_loss=val_metrics["loss"],
                       val_accuracy=val_metrics["accuracy"], learning_rate=lr,
                       train_seconds=train_seconds)
            history.append(row)
            writer.writerow(row)
            file.flush()
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_accuracy": best_accuracy}, trial_dir / "best.pt")
            print(f"{variant} seed={seed} epoch={epoch}/{args.epochs} "
                  f"train_loss={row['train_loss']:.4f} val_acc={best_accuracy:.2%} "
                  f"(best)", flush=True)
    save_training_curves(history, trial_dir / "training_curves.png", title=f"{variant}, seed {seed}")
    checkpoint = torch.load(trial_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device)
    result = dict(variant=variant, seed=seed, parameter_count=parameter_count,
                  initial_hash=fingerprint, best_epoch=checkpoint["epoch"],
                  best_val_accuracy=best_accuracy, test_accuracy=test_metrics["accuracy"],
                  test_loss=test_metrics["loss"],
                  mean_train_seconds=statistics.mean(row["train_seconds"] for row in history))
    (trial_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, history


def save_comparison(results, histories, output_dir, smoke_test):
    """Save per-run results, mean/sample-SD, paired differences, and validation curves."""
    summary = {"synthetic_smoke_test": smoke_test, "variants": {}, "paired_differences": []}
    for variant in VARIANTS:
        rows = [row for row in results if row["variant"] == variant]
        accuracies = [row["test_accuracy"] for row in rows]
        summary["variants"][variant] = {
            "test_accuracy_mean": statistics.mean(accuracies),
            "test_accuracy_sample_std": statistics.stdev(accuracies) if len(rows) > 1 else None,
            "runs": len(rows),
        }
    for seed in sorted({row["seed"] for row in results}):
        pair = {row["variant"]: row for row in results if row["seed"] == seed}
        assert pair["original"]["initial_hash"] == pair["single_skip"]["initial_hash"]
        assert pair["original"]["parameter_count"] == pair["single_skip"]["parameter_count"]
        summary["paired_differences"].append({
            "seed": seed,
            "single_skip_minus_original_percentage_points":
                100 * (pair["single_skip"]["test_accuracy"] - pair["original"]["test_accuracy"]),
        })
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    figure = Figure(figsize=(11, 4))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for variant in VARIANTS:
        trials = histories[variant]
        epochs = [row["epoch"] for row in trials[0]]
        for axis, metric, scale in zip(axes, ["val_accuracy", "val_loss"], [100, 1]):
            means, deviations = [], []
            for index in range(len(epochs)):
                values = [scale * trial[index][metric] for trial in trials]
                means.append(statistics.mean(values))
                deviations.append(statistics.stdev(values) if len(values) > 1 else 0)
            line, = axis.plot(epochs, means, label=variant)
            if len(trials) > 1:
                axis.fill_between(epochs, [m - s for m, s in zip(means, deviations)],
                                  [m + s for m, s in zip(means, deviations)],
                                  color=line.get_color(), alpha=0.15)
    axes[0].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 100))
    axes[1].set(title="Validation loss", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle("SYNTHETIC SMOKE TEST" if smoke_test else "Mixer skip connections: mean +/- sample SD")
    figure.tight_layout()
    figure.savefig(output_dir / "comparison.png", dpi=150)
    figure.clear()
    print(json.dumps(summary, indent=2))


def main(args):
    """Run paired trials; preserve partial per-trial artifacts if interrupted."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True)
    datasets = build_datasets(args)
    (output_dir / "experiment.json").write_text(json.dumps({
        **vars(args), "device": str(device), "torch_version": torch.__version__,
        "validation_indices": datasets[3],
    }, indent=2), encoding="utf-8")
    results = []
    histories = {variant: [] for variant in VARIANTS}
    for index, seed in enumerate(args.seeds):
        # Alternate ordering to reduce a systematic first-run timing advantage.
        order = list(VARIANTS) if index % 2 == 0 else list(reversed(VARIANTS))
        for variant in order:
            result, history = train_trial(variant, seed, args, datasets[:3], device, output_dir)
            results.append(result)
            histories[variant].append(history)
    save_comparison(results, histories, output_dir, args.smoke_test)
    print(f"Comparison saved to: {output_dir}")


if __name__ == "__main__":
    main(parse_args())
