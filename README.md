# PQD-1D-model
The repo is the official implementation for the paper: [Attention-enhanced residual networks for real-time multi-label power quality disturbance classification with fast iterative filtering](https://doi.org/10.1016/j.apenergy.2025.127233), containing:
1.	A simulated Power Quality Disturbance (PQD) database
2.	1D ResNet/MobileNet/DenseNet models for PQD classification
3.	Proposed multi-scale ResNet-efficient channel attention (ECA) PQD classification framework

## Updates
🚩 Feb 25, 2026: Code release for the ECA-enhanced ResNet framework for real-time multi-label PQD identification [(paper)](https://doi.org/10.1016/j.apenergy.2025.127233)

🚩 June 25, 2025: Synthetic PQD dataset and 1D ResNet/MobileNet/DenseNet models for PQD classification [(paper)](https://doi.org/10.1109/IECON55916.2024.10905211) are available.

## Introduction 
The growing integration of offshore wind energy into modern power grids introduces diverse PQDs arising from both offshore renewable dynamics and onshore grid events, underscoring the need for real-time, comprehensive PQ monitoring (see [here](https://doi.org/10.1016/j.rser.2023.114094) for more details).

Accordingly, we develop a lightweight classification framework that leverages fast iterative filtering (FIF), a multi-scale 1D ResNet, and ECA for superior predictive performance. The proposed architecture uses three parallel branches with varying kernel sizes to capture multi-level features from FIF-derived 1D sequences, enabling robust multi-label classification of overlapping PQ events.
![Framework](figures/framework.png)

## Synthetic dataset 
Synthetic voltage signals were generated in MATLAB per IEEE 1159–2019. Signals (0.2 s, 640 samples at 3.2 kHz) cover 20 events with multi-label annotations, including seven base disturbances and their combinations. Gaussian noise is added at SNRs of 20–50 dB to reflect practical conditions.

## Usage 

## Citation
Please cite our work if you find this repository helpful:

```bibtex
@article{shao2026attention,
  title={Attention-enhanced residual networks for real-time multi-label power quality disturbance classification with fast iterative filtering},
  author={Shao, Han and Henriques, Rui and Morais, Hugo and Tedeschi, Elisabetta},
  journal={Applied Energy},
  volume={406},
  pages={127233},
  year={2026},
  publisher={Elsevier}
}
