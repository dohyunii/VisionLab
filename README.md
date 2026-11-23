# Modular Computer Vision Experiment Framework

This is a **small, educational starter project**.


```bash
python -m pip install -r requirements.txt
python main.py --dataset cifar10 --arch resnet50 --method supervised --epochs 10 --batch_size 128
python main.py --help
```

## Optimizers

Select `--optimizer sgd` (the default) or `--optimizer adamw`.
Both use `--lr` and `--weight_decay`; `--momentum` applies only to SGD.
AdamW uses PyTorch's default betas `(0.9, 0.999)` and epsilon `1e-8`.
The default learning rate is still `0.1`, so set it explicitly for AdamW:

```bash
python main.py --arch mlpmixerS4 --head linear --optimizer adamw --lr 0.0003 --weight_decay 0.01
```

Optimizer selection is saved in `config.json` and the checkpoint configuration.
The existing milestone learning-rate schedule applies to either optimizer.

## Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dohyunii/VisionLab/blob/main/colab.ipynb)

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

## Mixer comparison

`compare_mixers.py` compares the standard Mixer (a skip around each mixing MLP)
with the custom Mixer (one skip around both mixing MLPs). The hypothesis is that
coupling the two mixing operations may change generalization and convergence;
the script does not assume either design is better.

The full experiment uses your CIFAR S/4 configuration for both: 8 blocks,
embedding width 256, token MLP width 128, channel MLP width 1024, and a linear
10-class classifier. Each seed starts both variants with identical weights,
verified by a saved SHA-256 fingerprint, and identical parameter counts.

Shared settings:

- CIFAR-10: 45,000 training images, 5,000 validation images; split seed 42.
- Training augmentation: random crop with padding 4 and horizontal flip.
- Validation/test: tensor conversion and normalization only, using separate
  dataset instances so validation never inherits random training transforms.
- AdamW, learning rate 0.0003, weight decay 0.01, batch size 128, 100 epochs.
- Learning rate multiplied by 0.1 after 50% and 75% of the epochs; no warmup.
- Paired training seeds 42, 43, and 44, with matched shuffle and augmentation RNGs.
- Best epoch selected using validation accuracy; its checkpoint is evaluated
  once on the official test set. Do not tune settings against these test scores.

This is a controlled baseline experiment, not a paper-reproduction recipe or
an optimized hyperparameter search. Seeds do not guarantee bitwise equality
across different devices or software versions.

Run this in Colab after installing dependencies and mounting Drive:

```python
!python -u compare_mixers.py \
    --data_dir /content/drive/MyDrive/VisionLab-data \
    --save_dir /content/drive/MyDrive/VisionLab-runs/mixer-comparison \
    --epochs 100 --seeds 42 43 44
```

This runs six training trials sequentially. It downloads CIFAR-10 only if the
shared cache is missing. Existing `main.py` training behavior is unchanged;
held-out evaluation is specific to this comparison script.

For a short execution check with synthetic data, smaller models, and no downloads:

```bash
python compare_mixers.py --smoke_test
```

The smoke test is explicitly labeled synthetic and provides no model-quality
evidence. It forces two epochs, one seed, batch size 4, and zero loader workers.

Each timestamped experiment contains:

```text
experiment.json             # Settings, device, and fixed validation indices
results.csv                 # One row per variant/seed
summary.json                # Test accuracy mean, sample SD, paired differences
comparison.png              # Validation curves: mean +/- sample SD across seeds
original_seed42/            # Similar folder for each variant/seed
    config.json             # Initial-weight fingerprint and parameter count
    history.csv             # Train/validation loss and accuracy, LR, train seconds
    training_curves.png
    best.pt                 # Validation-selected checkpoint
    result.json
```

Interpret positive `single_skip_minus_original_percentage_points` as higher
test accuracy for the custom model for that seed. Check whether differences
repeat across seeds and compare them with run-to-run variation; three seeds
alone do not establish statistical significance. Compare validation curves for
convergence/overfitting. Recorded train times include data loading and warmup
effects, so they are not a dedicated GPU performance benchmark. Resume support
is not included; per-trial histories and best checkpoints survive interruptions.

Implementation guide: `build_datasets()` creates the held-out split,
`build_loaders()` seeds each trial's data stream, `evaluate()` calculates
sample-weighted classification metrics without gradients, `train_trial()`
trains/selects/tests one model, and `save_comparison()` aggregates the paired
results and plots. The custom backbone is also available through `main.py` as
`--arch custommlpmixerS4`.
