import torch.nn.functional as F
from torch import nn


class Supervised(nn.Module):
    def __init__(
        self,
        backbone,
        feature_dim,
        num_classes,
        head="linear",
        head_hidden_dim=None,
        head_activation="relu"
    ):
        super().__init__()
        self.backbone = backbone
        if head == "linear":
            self.head = nn.Linear(feature_dim, num_classes)
        elif head == "mlp":
            hidden_dim = feature_dim if head_hidden_dim is None else head_hidden_dim
            activations = {"tanh": nn.Tanh, "gelu": nn.GELU, "relu": nn.ReLU}
            if head_activation not in activations:
                raise ValueError(f"Unknown head activation: {head_activation}")
            self.head = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                activations[head_activation](),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError(f"Unknown head: {head}")

    def forward(self, images):
        return self.head(self.backbone(images))

    def training_step(self, batch, return_logits=False):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        if return_logits:
            return loss, logits.detach()
        return loss
