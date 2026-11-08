import argparse
import json
import math
from datetime import datetime
from pathlib import Path
import torch
from architectures import ARCHITECTURES, load_arch
from datasets import DATASETS, load_dataset
from methods import load_method
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Modular computer-vision training")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="cifar10")
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), required=True)
    parser.add_argument("--method", choices=["supervised"], default="supervised")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./runs")
    return parser.parse_args()


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    for epoch in range(1, args.epochs + 1):
        method.train()
        total_loss = 0.0
        total_samples = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = method.training_step((images, labels))
            loss.backward()
            optimizer.step()

            size = images.size(0)
            total_loss += loss.item() * size
            total_samples += size
        scheduler.step()

        mean_loss = total_loss / total_samples
        print(f"Epoch {epoch}/{args.epochs}: train_loss={mean_loss:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "method_state_dict": method.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(args),
            },
            run_dir / "last.pt",
        )

    print(f"Saved checkpoint and config to: {run_dir}")

if __name__ == "__main__":
    main(parse_args())