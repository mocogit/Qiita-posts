#
# -*- coding: utf-8 -*-
#
#   kaggle_otto_xgb.py
#       date. 7/10/2017
#

import argparse
import numpy as np
import pandas as pd
from time import time

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import lightgbm as lgb

def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X)
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)


    return y, encoder


def make_submission(y_prob, ids, encoder, fname):
    with open(fname, 'w') as f:
        f.write('id,')
        f.write(','.join([str(i) for i in encoder.classes_]))
        f.write('\n')
        for i, probs in zip(ids, y_prob):
            probas = ','.join([i] + [str(p) for p in probs.tolist()])
            f.write(probas)
            f.write('\n')
    print('Wrote submission to file {}.'.format(fname))


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

    params['max_depth'] = 6
    params['learning_rate'] = 0.1   # alias of 'eta'
    params['objective'] = 'multi:softmax'
    params['n_estimators'] = 500
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

    params['num_leaves'] = 64 
    params['learning_rate'] = 0.1
    params['objective'] = 'multiclass'
    params['n_estimators'] = 500
    params['nthread'] = 8

    return params


def main(args):
    # load data
    X, labels = load_data('../data/train.csv', train=True)
    X, scaler = preprocess_data(X)
    y, encoder = preprocess_labels(labels)

    X_test, ids = load_data('../data/test.csv', train=False)
    X_test, _ = preprocess_data(X_test, scaler)

    # Stratified k-fold
    np.random.seed(201707)
    start = time()
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

    end = time()
    e_time = end - start
    # check prediction 
    print('Cross Validation resuls:')
    score_m = np.mean(scores)   # mean
    score_v = np.std(scores)    # standard deviation
    print('score = {:>8.4f} +/- {:>8.4f}\n'.format(score_m, score_v))
    print('Elapse time = {:>8.2f} s'.format(e_time))

    y_pred = gb_clf.predict(X_val)
    y_pred_proba = gb_clf.predict_proba(X_val)
    accu = accuracy_score(y_val, y_pred)
    mlogloss = log_loss(y_val, y_pred_proba)
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
