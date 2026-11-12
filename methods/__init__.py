from .supervised import Supervised

METHODS = {"supervised": Supervised}


def load_method(
    name,
    backbone,
    feature_dim,
    num_classes,
    head="linear",
    head_hidden_dim=None,
    head_activation="relu",
):
    try:
        method_class = METHODS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown learning method: {name}") from exc

    return method_class(
        backbone,
        feature_dim,
        num_classes,
        head=head,
        head_hidden_dim=head_hidden_dim,
        head_activation=head_activation,
    )
