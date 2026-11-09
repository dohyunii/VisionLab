# Modular Computer Vision Experiment Framework

This is a **small, educational starter project**.


```bash
python -m pip install -r requirements.txt
python main.py --dataset cifar10 --arch resnet50 --method supervised --epochs 10 --batch_size 128
python main.py --help
```

## Google Colab

```python
!git clone https://github.com/dohyunii/VisionLab
%cd /content/VisionLab
!pip install -r requirements.txt
```

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!python main.py --dataset cifar10 --arch resnet18 --method supervised \
    --epochs 10 --batch_size 128 \
    --save_dir /content/drive/MyDrive/cv-runs
```

## Saved outputs

By default, datasets are downloaded once into a shared `data/` folder inside
`--save_dir`. Experiments reuse this folder instead of keeping a copy per run.
An explicit `--data_dir` overrides this location; use `--data_dir ./data` to reuse
datasets downloaded with the previous default.

```text
runs/
├── data/                       # Shared CIFAR files
└── <dataset>_<arch>_<method>_<timestamp>/
    ├── config.json             # Arguments, including the resolved dataset path
    ├── last.pt                 # Latest model, optimizer, scheduler, and metrics
    ├── history.csv             # One row per completed epoch
    └── training_curves.png     # Training loss and accuracy curves
```

`history.csv` records `epoch`, `train_loss`, `train_accuracy` (a fraction from 0 to 1),
and the `learning_rate` used during that epoch. The console and plot display
accuracy as a percentage. History and plots are updated after every epoch.
Metrics describe the augmented training batches; no validation or test accuracy
is computed.

The Colab command above stores both the shared dataset and experiment outputs
in Google Drive. The latest checkpoint is overwritten each epoch; training
resumption is not implemented yet.
