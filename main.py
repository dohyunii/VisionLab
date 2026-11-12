import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
import torch
from architectures import ARCHITECTURES, load_arch
from datasets import DATASETS, load_dataset
from methods import load_method
from utils import save_training_curves, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Modular computer-vision training")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="cifar10")
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), required=True)
    parser.add_argument("--method", choices=["supervised"], default="supervised")
    parser.add_argument("--head", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--head_hidden_dim", type=int, default=None)
    parser.add_argument("--head_activation", choices=["tanh", "gelu", "relu"], default="relu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Dataset directory (default: a shared data/ folder inside --save_dir)",
    )
    parser.add_argument("--save_dir", type=str, default="./runs")
    return parser.parse_args()


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    data_dir = Path(args.data_dir) if args.data_dir is not None else Path(args.save_dir) / "data"
    args.data_dir = str(data_dir.resolve())
    print(f"Shared dataset directory: {args.data_dir}")

    train_loader, num_classes = load_dataset(
        name=args.dataset,
        batch_size=args.batch_size,
        method=args.method,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    backbone, feature_dim = load_arch(args.arch)
    method = load_method(
        name=args.method,
        backbone=backbone,
        feature_dim=feature_dim,
        num_classes=num_classes,
        head=args.head,
        head_hidden_dim=args.head_hidden_dim,
        head_activation=args.head_activation,
    ).to(device)

    optimizer = torch.optim.SGD(
        method.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)],
        gamma=0.1
    )

    run_name = (
        f"{args.dataset}_{args.arch}_{args.method}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    run_dir = Path(args.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    history = []
    history_path = run_dir / "history.csv"
    history_fields = ["epoch", "train_loss", "train_accuracy", "learning_rate"]
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        csv.DictWriter(history_file, fieldnames=history_fields).writeheader()

    for epoch in range(1, args.epochs + 1):
        method.train()
        learning_rate = optimizer.param_groups[0]["lr"]
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss, logits = method.training_step((images, labels), return_logits=True)
            loss.backward()
            optimizer.step()

            size = images.size(0)
            total_loss += loss.item() * size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += size
        scheduler.step()

        mean_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "train_accuracy": accuracy,
            "learning_rate": learning_rate,
        }
        history.append(epoch_metrics)
        with history_path.open("a", newline="", encoding="utf-8") as history_file:
            csv.DictWriter(history_file, fieldnames=history_fields).writerow(epoch_metrics)
        save_training_curves(
            history, run_dir / "training_curves.png",
            title=f"{args.arch} · {args.dataset.upper()} · {args.method.capitalize()}",
        )
        print(
            f"Epoch {epoch}/{args.epochs}: train_loss={mean_loss:.4f} "
            f"train_accuracy={accuracy:.2%} lr={learning_rate:g}"
        )

        torch.save(
            {
                "epoch": epoch,
                "method_state_dict": method.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(args),
                "metrics": epoch_metrics,
            },
            run_dir / "last.pt",
        )

    print(f"Saved checkpoint, config, history, and plots to: {run_dir}")

if __name__ == "__main__":
    main(parse_args())
