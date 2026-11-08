import torch.nn.functional as F
from torch import nn


class Supervised(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, images):
        return self.head(self.backbone(images))

    def training_step(self, batch):
        images, labels = batch
        logits = self(images)
        return F.cross_entropy(logits, labels)
