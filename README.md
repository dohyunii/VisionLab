# Modular Computer Vision Experiment Framework

This is a **small, educational starter project**.


```bash
python -m pip install -r requirements.txt
python main.py --dataset cifar10 --arch resnet18 --method supervised --epochs 10 --batch_size 128
python main.py --dataset cifar10 --arch densenet121 --method rotnet --epochs 10 --batch_size 64
python main.py --help
```

## Google Colab

```python
!git clone https://github.com/dohyunii/VisionLab
%cd /content/cv-experiment-framework
!pip install -r requirements.txt
```

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!python main.py --dataset cifar10 --arch resnet18 --method supervised \
    --epochs 10 --batch_size 128 --data_dir /content/data \
    --save_dir /content/drive/MyDrive/cv-runs
```