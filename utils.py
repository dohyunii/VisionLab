import random
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_training_curves(history, path, title=None):
    """Save training loss and accuracy without requiring a display."""
    figure = Figure(figsize=(10, 4))
    FigureCanvasAgg(figure)
    loss_axis, accuracy_axis = figure.subplots(1, 2)
    epochs = [row["epoch"] for row in history]
    losses = [row["train_loss"] for row in history]
    accuracies = [100 * row["train_accuracy"] for row in history]
    loss_axis.plot(epochs, losses, linewidth=1.5, marker="")
    loss_axis.set(xlabel="Epoch", ylabel="Loss", title="Training loss")
    accuracy_axis.plot(
        epochs, accuracies, linewidth=1.5, marker=""
    )
    accuracy_axis.set(
        xlabel="Epoch", ylabel="Accuracy (%)", title="Training accuracy", ylim=(0, 100)
    )
    # History records the LR used in each epoch. A transition in epoch N means
    # the scheduler changed the LR at the end of the preceding recorded epoch.
    lr_changes = [
        previous["epoch"]
        for previous, current in zip(history, history[1:])
        if previous.get("learning_rate") is not None
        and current.get("learning_rate") is not None
        and previous["learning_rate"] != current["learning_rate"]
    ]
    for axis in (loss_axis, accuracy_axis):
        axis.grid(alpha=0.3)
        for index, epoch in enumerate(lr_changes):
            axis.axvline(
                epoch, color="0.45", linestyle="--", linewidth=1,
                label="LR change (end of epoch)" if index == 0 else "_nolegend_",
            )
        if lr_changes:
            axis.legend(fontsize=8)

    loss_axis.annotate(
        f"{losses[-1]:.4f}", (epochs[-1], losses[-1]),
        xytext=(-6, 12), textcoords="offset points", ha="right", fontsize=9,
    )
    accuracy_axis.annotate(
        f"{accuracies[-1]:.2f}%", (epochs[-1], accuracies[-1]),
        xytext=(-6, -16 if accuracies[-1] >= 85 else 12),
        textcoords="offset points", ha="right", fontsize=9,
    )
    if title:
        figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.93) if title else (0, 0, 1, 1))
    figure.savefig(path, dpi=150)
    figure.clear()
