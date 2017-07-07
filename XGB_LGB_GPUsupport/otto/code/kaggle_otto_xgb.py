#
# -*- coding: utf-8 -*-
#
#   kaggle_otto_xgb.py
#       date. 6/27/2017
#

import numpy as np
import pandas as pd
from time import time
np.random.seed(201707)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb

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


def lgb_set_param():
    '''
      XGBoost (classifier) parameters
    '''
    params = {}
    params['updater'] = 'grow_histmaker,prune'
    params['max_depth'] = 6
    params['learning_rate'] = 0.1
    params['objective'] = 'multi:softmax'
    params['n_estimators'] = 500
    params['n_jobs'] = -1
    params['device'] = 'cpu'

    return params


if __name__ == '__main__':
    X, labels = load_data('../data/train.csv', train=True)
    X, scaler = preprocess_data(X)
    y, encoder = preprocess_labels(labels)

    X_test, ids = load_data('../data/test.csv', train=False)
    X_test, _ = preprocess_data(X_test, scaler)

    # X : (61878, 93), y : 0..8

    # Stratified k-fold
    start = time()
    skf = StratifiedKFold(n_splits=5)
    scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train = X[train_idx]
        y_train = y[train_idx]

        params = lgb_set_param()
        xgb_clf = xgb.XGBClassifier(**params)

        X_val = X[test_idx]
        y_val = y[test_idx]

        xgb_clf.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mlogloss'
            # early_stopping_rounds=10 ... not apply for benchmark
        )
        score = xgb_clf.score(X_val, y_val)
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

    y_pred = xgb_clf.predict(X_val)
    y_pred_proba = xgb_clf.predict_proba(X_val)
    accu = accuracy_score(y_val, y_pred)
    mlogloss = log_loss(y_val, y_pred_proba)
    print('accuracy = {:>8.4f}'.format(accu))
    print('multi-class logloss = {:>8.4f}'.format(mlogloss))
