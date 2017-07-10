# -*- coding: utf-8 -*-
#
#   mnist_xgb.py
#       date. 7/10/2017
#

import argparse
import time
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from mnist_loader import load_data

def mnist_load_data(dirn='../data'):
    '''
      Load MNIST data from files
    '''
    X_train, y_train = load_data(dirn, subset='train')
    X_test, y_test = load_data(dirn, subset='test')

    # reshape image data to flat
    X_train = X_train.reshape([-1, 28*28])
    X_test = X_test.reshape([-1, 28*28])

    # scaling image data from [0 .. 255] to [0.0 .. 1.0]
    X_train = X_train / 255.
    X_test = X_test / 255.

    return X_train, X_test, y_train, y_test


def xgb_set_param(args):
    '''
      XGBoost (classifier) parameters
    '''
    params = {}
    if args.cpu:
        params['device'] = 'cpu'
        params['updater'] = 'grow_histmaker,prune'
    else:   # gpu
        params['device'] = 'gpu'
        params['gpu_id'] = 0
        params['updater'] = 'grow_gpu_hist'

    params['max_depth'] = 5
    params['learning_rate'] = 0.2
    params['objective'] = 'multi:softmax'
    params['n_estimators'] = 200
    params['n_jobs'] = -1

    return params

def lgb_set_param(args):
    '''
      LightGBM (classifier) parameters
    '''
    params = {}
    if args.cpu:
        params['device'] = 'cpu'
    else:   # gpu
        params['device'] = 'gpu'
        params['max_bin'] = 15
        params['gpu_use_dp'] = False

    params['num_leaves'] = 32
    params['learning_rate'] = 0.2
    params['objective'] = 'multiclass'
    params['n_estimators'] = 200
    params['nthread'] = 8
    
    return params

def main(args):
    # load data
    X, X_test, y, y_test = mnist_load_data()

     # Stratified K-fold
    np.random.seed(201707)
    print('Stratified K-fold process:')
    start = time.time()
    skf = StratifiedKFold(n_splits=5)
    scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train = X[train_idx]
        y_train = y[train_idx]

        if args.xgboost:
            params = xgb_set_param(args)
            gb_clf = xgb.XGBClassifier(**params)
            eval_str = 'mlogloss'
        if args.lightgbm:
            params = lgb_set_param(args)
            gb_clf = lgb.LGBMClassifier(**params)
            eval_str = 'multi_logloss'

        X_val = X[test_idx]
        y_val = y[test_idx]

        gb_clf.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=eval_str
        )
        score = gb_clf.score(X_val, y_val)
        print('score = ', score)
        scores.append(score)

    end = time.time()
    e_time = end - start
    # check prediction 
    print('Cross Validation resuls:')
    score_m = np.mean(scores)   # mean
    score_v = np.std(scores)    # standard deviation
    print('score = {:>8.4f} +/- {:>8.4f}\n'.format(score_m, score_v))
    print('Elapse time = {:>8.2f} s'.format(e_time))

    y_pred = gb_clf.predict(X_test)
    y_pred_proba = gb_clf.predict_proba(X_test)
    accu = accuracy_score(y_test, y_pred)
    mlogloss = log_loss(y_test, y_pred_proba)
    print('accuracy = {:>8.4f}'.format(accu))
    print('multi-class logloss = {:>8.4f}'.format(mlogloss))


if __name__ == '__main__':
    # parse argument
    parser = argparse.ArgumentParser()
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("--cpu", action="store_true",
                    help="no GPU support")
    group1.add_argument("--gpu", action="store_true",
                    help="GPU support")

    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument("--xgboost", "-X", action="store_true",
                    help="XGBoost classifier")
    group2.add_argument("--lightgbm", "-L", action="store_true",
                    help="LightGBM classifier")
    args = parser.parse_args()

    if args.cpu:
        print('selected cpu option.')
    elif args.gpu:
        print('selected gpu option.')
    else:
        raise ValueError('no spec: [cpu | gpu]')

    if args.xgboost:
        print('selected XGBoost.')
    elif args.lightgbm:
        print('selected LightGBM.')
    else:
        raise ValueError('no spec: [xgboost | lightgbm]')

    main(args)
