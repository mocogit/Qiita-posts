# -*- coding: utf-8 -*-
#
#   mnist_lgb.py
#       date. 7/9/2017
#

import time
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from mnist_loader import load_data

def mnist_load_data():
    '''
      Load MNIST data from files
    '''
    dirn = '../data'
    X_train, y_train = load_data(dirn, subset='train')
    X_test, y_test = load_data(dirn, subset='test')

    # reshape image data to flat
    X_train = X_train.reshape([-1, 28*28])
    X_test = X_test.reshape([-1, 28*28])

    # scaling image data from [0 .. 255] to [0.0 .. 1.0]
    X_train = X_train / 255.
    X_test = X_test / 255.

    return X_train, X_test, y_train, y_test


def lgb_set_param():
    '''
      LightGBM (classifier) parameters
    '''
    params = {}
    params['num_leaves'] = 32
    params['learning_rate'] = 0.2
    params['objective'] = 'multiclass'
    params['n_estimators'] = 200
    params['nthread'] = 8

    return params


if __name__ == '__main__':
    np.random.seed(201707)
    X, X_test, y, y_test = mnist_load_data()

    # Stratified K-fold
    print('Stratified K-fold process:')
    start = time.time()
    skf = StratifiedKFold(n_splits=5)
    scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train = X[train_idx]
        y_train = y[train_idx]

        params = lgb_set_param()
        lgb_clf = lgb.LGBMClassifier(**params)

        X_val = X[test_idx]
        y_val = y[test_idx]

        lgb_clf.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss'
            # early_stopping_rounds=10
        )
        score = lgb_clf.score(X_val, y_val)
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

    y_pred = lgb_clf.predict(X_test)
    y_pred_proba = lgb_clf.predict_proba(X_test)
    accu = accuracy_score(y_test, y_pred)
    mlogloss = log_loss(y_test, y_pred_proba)
    print('accuracy = {:>8.4f}'.format(accu))
    print('multi-class logloss = {:>8.4f}'.format(mlogloss))
