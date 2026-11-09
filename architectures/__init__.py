from .resnet import resnet18, resnet50
from .preactresnet import preactresnet18, preactresnet50
from .densenet import densenet121
from .fractalnet import fractalnet20, fractalnet40

ARCHITECTURES = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "preactresnet18": preactresnet18,
    "preactresnet50": preactresnet50,
    "densenet121": densenet121,
    "fractalnet20": fractalnet20,
    "fractalnet40": fractalnet40
}

def load_arch(name):
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {name}")

    model = ARCHITECTURES[name]()
    return model, model.feature_dim