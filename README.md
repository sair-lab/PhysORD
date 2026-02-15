# PhysORD: A Neuro-Symbolic Approach for Physics-infused Motion Prediction in Off-road Driving

This repository provides the official implementation for paper: 

[**Zhipeng Zhao**](https://github.com/Zhaozhpe), [**Bowen Li**](https://github.com/Jaraxxus-Me), [**Yi Du**](https://github.com/Inoriros), [**Taimeng Fu**](https://github.com/FuTaimeng), and [**Chen Wang**](https://github.com/wang-chen), "[PhysORD: A Neuro-Symbolic Approach for Physics-infused Motion Prediction in Off-road Driving](https://arxiv.org/abs/2404.01596)." IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024

## Prerequisites

- Python 3.10.13
- PyTorch 2.0.1
- PyPose 0.6.7

## Dataset
- This project utilizes the [TartanDrive](https://github.com/castacks/tartan_drive) dataset. Follow the instructions in its [repository](https://github.com/castacks/tartan_drive?tab=readme-ov-file#create-traintest-split) to create the `train`, `test-easy` and `test-hard` sets.
- `test-easy` is used for validation during training, and `test-hard` for model evaluation.
- We also provide pre-processed data with 20-step and 5-step sequences for quick reproduction. You can download them into the [data](data) folder.
```
# 20-step
wget -P data/ https://github.com/sair-lab/PhysORD/releases/download/data/train_val_easy_507_step20.pt

# 5-step
wget -P data/ https://github.com/sair-lab/PhysORD/releases/download/data/train_val_easy_507_step5.pt
```

## Reproduce Guide
To reproduce our result in the paper, you can follow the the steps below.

### Train
- You need to set the size of the training data with `--train_data_size` (from 1 to 507), and the number of training steps with `--timesteps`.
- You can specify the prepared data directory by `--preprocessed_data_dir`, and the the directory for saving the model by `--save_dir`.
```
python train.py
```

### Evalution
- Specify the path to the evaluation data with `--eval_data_fp`, and the test timesteps with `--timesteps`.
- You can also set the sample intervals of the data with `--test_sample_interval`.
- We provide a 20-step and a 5-step pretrained models for quick evaluation - see the folder [pretrained](pretrained) for both models.
```
python test.py
```


## Citation
```
@inproceedings{zhao2024physord,
  title = {{PhysORD}: A Neuro-Symbolic Approach for Physics-infused Motion Prediction in Off-road Driving},
  author = {Zhao, Zhipeng and Li, Bowen and Du, Yi and Fu, Taimeng and Wang, Chen},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year = {2024},
  pages = {11670--11677}
}
```
