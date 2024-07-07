Welcome to **Spatial-Temporal Graph Neural Networks** repository.

This repository contains the implementation of MTGNN-TAttLA model described in the paperr [Deep learning-based spatial-temporal graph neural networks for price movement classification in crude oil and precious metal markets](https://doi.org/10.1016/j.mlwa.2024.100552)  paper.

## Table of Contents
- [Introduction](#introduction)
- [Setup](#setup)
- [Training](#training)
- [Citation](#Citation)

## Introduction

The model aims to classify future direction of finanical market trends using a novel graph-based nueral network that combines spatial representations and temporal representations of the input data. 

## Setup

### Prerequisites

- Python 3.12 or higher

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Parisa-Foroutan/Timeseries-forecasting-STGNN.git
    cd Timeseries-forecasting-STGNN
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Training

To train the model, run the following command:

```bash
python src/train.py --data_path data/ --epochs 100 --batch_size 32 --learning_rate 0.001
```

## Citation

```bash
Parisa Foroutan, Salim Lahmiri,
Deep learning-based spatial-temporal graph neural networks for price movement classification in crude oil and precious metal markets,
Machine Learning with Applications,
Volume 16, 2024, 100552, ISSN 2666-8270,
https://doi.org/10.1016/j.mlwa.2024.100552.
```
