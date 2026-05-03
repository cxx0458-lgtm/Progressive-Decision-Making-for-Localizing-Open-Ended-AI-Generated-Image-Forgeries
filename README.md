Progressive Decision-Making for Localizing Open-Ended AI-Generated Image Forgeries
========

This repository contains the PyTorch implementation of our paper: **"Progressive Decision-Making for Localizing Open-Ended AI-Generated Image Forgeries"**.

This work focuses on pixel-level localization of open-ended AI-generated image forgeries. The code is developed based on the Mesorch / IMDLBenCo-style image manipulation localization framework, and introduces a progressive decision-making mechanism to improve localization robustness for AI-generated manipulation scenarios.

**Note:**  
This project is built upon [IMDLBenCo](https://github.com/scu-zjz/IMDLBenCo) and Mesorch-style training/testing pipelines. For dataset format, environment details, and additional framework-related issues, please also refer to the original IMDLBenCo/Mesorch repositories.

## Environment
<details>
<summary><b>Click to expand</b></summary>

The environment used in our experiments was mainly configured by installing IMDLBenCo first, followed by several additional dependencies required by the progressive refinement module.

A typical setup is:

```bash
conda create -n mesorch python=3.10
conda activate mesorch

pip install torch torchvision
pip install imdlbenco
pip install "numpy<2"
```

Then install the additional packages required by the model, such as Mamba-related libraries and other common dependencies:

```bash
pip install mamba-ssm causal-conv1d
pip install opencv-python pillow tqdm tensorboard albumentations scikit-learn
```

The exact package versions may depend on your CUDA and PyTorch versions. If installation of `mamba-ssm` or `causal-conv1d` fails, please install the version compatible with your local CUDA/PyTorch environment.

</details>

## File Structure

```plaintext
Progressive-Decision-Making-for-Localizing-Open-Ended-AI-Generated-Image-Forgeries/
├── progressive_mesorch.py       # Proposed model
├── train.py                     # Training script
├── test.py                      # Testing script
├── balanced_dataset1.json       # Training configuration for Protocol I
├── balanced_dataset2.json       # Training configuration for Protocol II
├── test_dataset1.json           # Testing configuration for Protocol I
├── test_dataset2.json           # Testing configuration for Protocol II
├── finetune_protocol1.sh        # Fine-tuning script for Protocol I
├── finetune_protocol2.sh        # Fine-tuning script for Protocol II
├── test_mesorch_f1.sh           # Testing script for standard F1-score
├── test_mesorch_pf1.sh          # Testing script for Permute F1-score
├── LICENSE
└── README.md
```

## Dataset Preparation
<details>
<summary><b>Click to expand</b></summary>

The dataset paths are specified by JSON files.

- `balanced_dataset1.json`: training configuration for Protocol I.
- `balanced_dataset2.json`: training configuration for Protocol II.
- `test_dataset1.json`: testing configuration for Protocol I.
- `test_dataset2.json`: testing configuration for Protocol II.

Please modify the paths in these JSON files according to your local dataset location.

A typical JSON format is:

```json
{
  "DatasetName": "/path/to/dataset/or/json/file"
}
```

Each image should have a corresponding binary ground-truth mask for pixel-level manipulation localization.

</details>

## Training Instructions
<details>
<summary><b>Click to expand</b></summary>

Before training, please check and modify the dataset paths and checkpoint paths in the shell scripts.

### Protocol I

Protocol I uses `balanced_dataset1.json` as the training configuration.

```bash
sh finetune_protocol1.sh
```

### Protocol II

Protocol II uses `balanced_dataset2.json` as the training configuration. This setting introduces AI-generated manipulation data during training.

```bash
sh finetune_protocol2.sh
```

</details>

## Testing Instructions
<details>
<summary><b>Click to expand</b></summary>

Before testing, please modify the checkpoint path and testing dataset path in the corresponding shell script.

### Standard F1-score

```bash
sh test_mesorch_f1.sh
```

### Permute F1-score

```bash
sh test_mesorch_pf1.sh
```

The testing dataset can be switched by modifying `--test_data_json` in the shell script:

- `test_dataset1.json` for Protocol I.
- `test_dataset2.json` for Protocol II.

</details>

## Model

The proposed model is implemented in:

```plaintext
progressive_mesorch.py
```

The registered model name is:

```plaintext
ProgressiveMesorch
```

Please make sure the model name in the shell scripts is consistent with the registered name.

## Checkpoints

Pretrained checkpoints are not included in this repository due to file size limitations.

Please place your checkpoints in the corresponding checkpoint directory and modify `--checkpoint_path` in the testing script.

## Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@article{hou2026progressive,
  title={Progressive Decision-Making for Localizing Open-Ended AI-Generated Image Forgeries},
  author={Hou, Jingyi and Chen, Xiaoxia and Zhou, Leyu and Wang, Zhichuang and Liu, Zhijie},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2026}
}
```

The citation information will be updated after publication.

## License

This project is released under the MIT License.
