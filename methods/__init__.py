from .supervised import Supervised

_METHODS = {"supervised": Supervised}


def load_method(name, backbone, feature_dim, num_classes):
    try:
        method_class = _METHODS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown learning method: {name}") from exc
    return method_class(backbone, feature_dim, num_classes)
