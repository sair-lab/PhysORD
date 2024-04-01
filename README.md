# PhysORD: A Neuro-Symbolic Approach for Physics-infused Motion Prediction in Off-road Driving

[**Zhipeng Zhao**](https://github.com/Zhaozhpe), [**Bowen Li**](https://github.com/Jaraxxus-Me), [**Yi Du**](https://github.com/Inoriros), [**Taimeng Fu**](https://github.com/FuTaimeng), and [**Chen Wang***](https://github.com/wang-chen)

This repository provides the official implementation of our paper, "PhysORD: A Neuro-Symbolic Approach for Physics-infused Motion Prediction in Off-road Driving" [[PDF]()].

## Prerequisites

- Python 3.10.13
- PyTorch 2.0.1
- PyPose 0.6.7

## Dataset
- This project utilizes the [TartanDrive](https://github.com/castacks/tartan_drive) dataset. Follow the instructions in its [repository](https://github.com/castacks/tartan_drive?tab=readme-ov-file#create-traintest-split) to create the `train`, `test-easy` and `test-hard` sets.
- `test-easy` is used for validation during training, and `test-hard` for model evaluation.
- We also provide pre-processed data for quick reproduction. Download the [train_val_easy_507_step20.pt / ...step5.pt](https://drive.google.com/drive/folders/16PX9j6SUU8_LB0vq5wj31WWVUY49cR8l?usp=sharing) file into the [data](data) folder.

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
python eval.py
```


## Citation
If you find our research helpful for your work, please consider starring this repo and citing our paper.



## License

This project is available under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
