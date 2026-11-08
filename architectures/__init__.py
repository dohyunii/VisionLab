from .resnet import ResNet
from .preactresnet import PreActResNet
from .densenet import DenseNet

ARCHITECTURES = {
    "resnet": ResNet,
    "preactresnet": PreActResNet,
    "densenet": DenseNet,
}

def load_arch(name):
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {name}")

    model = ARCHITECTURES[name]()
    return model, model.feature_dim