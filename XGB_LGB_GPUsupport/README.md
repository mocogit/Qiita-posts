# XGBoost / LightGBM benchmark with Kaggle-Otto, MNIST classification

I checked the condition of XGBoost / LightGBM when GPU support is enable.

The result is shown in [my_benchmark_data.md](https://github.com/tomokishii/Qiita-posts/blob/master/XGB_LGB_GPUsupport/my_benchmark_data.md) .


Benchmark trial how to:
1. create "data" directory.
2. prepare datasets.
    - Kaggle-Otto: store "train.csv" and "test.csv" from [Kaggle Otto](https://www.kaggle.com/c/otto-group-product-classification-challenge) .
    - MNIST: store "train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz" from MNIST dataset web site.
3. `python kaggle_otto_clf.py [--cpu | --gpu] [--xgboost | --lightgbm]` or   
`python mnist_clf.py [--cpu | --gpu] [--xgboost | --lightgbm]`
    in "otto" / "mnist" directory.
