# My benchmark data

## Hardware condition

- CPU: Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz
- Memory size: 15GiB
- GPU: Nvidia Geforece 1080ti


## Software condition

- Linux, Ubuntu 16.04-LTS
- Graphic Driver Version: 378.13 
- GPGPU library: CUDA 8.0

## Kaggle Otto Classification Results

| dataset  |  classifier lib. |  GPU support  | time *1)  | accurarcy  | mlogloss  |
|:--------:|:----------------:|:-----------:|------:|:----------:|:---------:|
| kaggle otto | xgboost       |  no (CPU)   | 1299.49 s | 0.816   | 0.475     |
| kaggle otto | xgboost       |  yes (GPU)  | 340.03 s | 0.814   | 0.475    |
| kaggle otto | lightgbm      | no (CPU)    | 120.10 s | 0.825   | 0.482    |
| kaggle otto | lightgbm      | yes (GPU)   | 116.42 s | 0.824   | 0.486   |

*1) Elapse time for 5-folds cross validation process


## MNIST Classification Results

| dataset  |  classifier lib. |  GPU support  | time *1) | accurarcy  | mlogloss  |
|:--------:|:----------------:|:-----------:|------:|:----------:|:---------:|
| MNIST | xgboost       |  no (CPU)   | 3882.01 s | 0.978  | 0.068 |
| MNIST | xgboost       |  yes (GPU)  | 1027.71 s | 0.977   | 0.069   |
| MNIST | lightgbm      | no (CPU)    | 255.18 s | 0.976  | 0.078   |
| MNIST | lightgbm      | yes (GPU)   |  87.80 s | 0.976  | 0.078  |

*1) Elapse time for 5-folds cross validation process

