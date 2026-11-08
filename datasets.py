from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

STATS = {
    "cifar10": ((0.491, 0.482, 0.447), (0.247, 0.244, 0.262)),
    "cifar100": ((0.507, 0.487, 0.441), (0.268, 0.257, 0.276)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
DATASETS = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
}


def load_dataset(name, batch_size, method, data_dir="./data", num_workers=2, pin_memory=False):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")

    mean, std = STATS[name]
    operations = []
    if method == "supervised":
        operations.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
    operations.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    transform = transforms.Compose(operations)

    dataset = DATASETS[name](root=str(Path(data_dir)), train=True, download=True, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, len(dataset.classes)