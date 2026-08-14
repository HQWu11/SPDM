# SPDM: A Structure-Preserving Diffusion Model for Point Cloud Upsampling

This repository provides the core implementation of **SPDM**, a diffusion-based framework for point cloud upsampling. SPDM is designed to improve structural preservation during progressive denoising by introducing density-aware geometric priors and sustained cross-branch feature interaction.

## Release Status

The current release provides the **core implementation of SPDM** and the **recommended experimental environment** used in our study.

Additional auxiliary scripts, experiment configurations, pretrained models, and detailed reproduction instructions are being organized and will be further completed upon publication.

## Requirements

The following environment is recommended:

- Ubuntu 18.04 or later
- Python 3.7
- CUDA 11.1
- PyTorch 1.9.1 + CUDA 11.1
- torchvision 0.10.1 + CUDA 11.1
- torchaudio 0.9.1
- PyTorch3D 0.6.1
- NVIDIA RTX 3090 GPU

## Environment Setup

Create and activate a Conda environment:

```bash
conda create -n spdm python=3.7 -y
conda activate spdm
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

### PyTorch3D

PyTorch3D 0.6.1 should be installed separately using a package compatible with Python 3.7, CUDA 11.1, and PyTorch 1.9.1.

The package used in our environment is:

```text
pytorch3d-0.6.1-py37_cu111_pyt191.tar.bz2
```

It can be installed locally with:

```bash
conda install /path/to/pytorch3d-0.6.1-py37_cu111_pyt191.tar.bz2
```

## Repository Structure

```text
SPDM/
├── models/          # Core network components
├── pointnet2/       # Point-cloud processing, diffusion utilities, and related code
└── exp_configs/     # Experimental configurations
```

## Notes

This repository is currently an initial public release focused on the core implementation and environment configuration of SPDM. The remaining supporting files and detailed usage instructions will be added as the code package is further organized.

## Citation

Citation information will be added upon publication.
