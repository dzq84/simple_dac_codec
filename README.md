# simple_dac_codec

This repository provides a simplified implementation of the DAC codec from the original repository [descript-audio-codec](https://github.com/descriptinc/descript-audio-codec/). The original DAC codec uses `descript-audiotools`, which made training and debugging slightly inconvenient. Additionally, the original DAC codec encountered some issues when using DDP training. In this version, the model has been refactored to use PyTorch Lightning, making DDP training much easier and more efficient.

## Getting Started

To get started, please follow the steps below:

### 1. Clone the repository

```
git clone https://github.com/dzq84/simple_dac_codec.git
cd simple_dac_codec
```

### 2. Install the required dependencies

```
pip install -r requirements.txt
```

### 3. Train your own codec model

```
python train.py --config config/config.yaml
```

## Citation

If you use this repository in your work, please cite：

```
@article{kumar2024high,
  title={High-fidelity audio compression with improved rvqgan},
  author={Kumar, Rithesh and Seetharaman, Prem and Luebs, Alejandro and Kumar, Ishaan and Kumar, Kundan},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```