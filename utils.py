import random
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_training_curves(history, path):
    """Save training loss and accuracy without requiring a display."""
    figure = Figure(figsize=(10, 4))
    FigureCanvasAgg(figure)
    loss_axis, accuracy_axis = figure.subplots(1, 2)
    epochs = [row["epoch"] for row in history]
    loss_axis.plot(epochs, [row["train_loss"] for row in history], linewidth=1.5)
    loss_axis.set(xlabel="Epoch", ylabel="Loss", title="Training loss")
    accuracy_axis.plot(
        epochs, [100 * row["train_accuracy"] for row in history], linewidth=1.5
    )
    accuracy_axis.set(
        xlabel="Epoch", ylabel="Accuracy (%)", title="Training accuracy", ylim=(0, 100)
    )
    for axis in (loss_axis, accuracy_axis):
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    figure.clear()
