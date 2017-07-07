# -*- coding: utf-8 -*-
#
#   mnist_xgb_cpu.py
#       date. 7/8/2017
#

import time
import numpy as np
import xgboost as xgb
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.linear_model import LogisticRegression

def load_data():
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


def xgb_gridsearch(X_train, X_test, y_train, y_test, n_folds=5):
    '''
      Base analysis process by XGBoost (Grid Search)
    '''
    param_grid = {
        'max_depth': [3, 4, 5], 
        'learning_rate': [0.1, 0.2],
        'n_estimators': [100]  }
    xgbclf = xgb.XGBClassifier()

    # Run Grid Search process
    fit_params = {'eval_metric': 'mlogloss',
                'verbose': False,
                'early_stopping_rounds': 10,
                'eval_set': [(X_test, y_test)]}
 
    gs_clf = GridSearchCV(xgbclf, param_grid, 
                          n_jobs=1, cv=n_folds,
                          fit_params=fit_params,
                          scoring='accuracy')
    gs_clf.fit(X_train, y_train)

    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    print('score:', score)
    for param_name in sorted(best_parameters.keys()):
        print('%s: %r' % (param_name, best_parameters[param_name]))

    xgbclf_best = xgb.XGBClassifier(**best_parameters)
    xgbclf_best.fit(X_train, y_train)
    y_pred_train = xgbclf_best.predict_proba(X_train)
    y_pred_test = xgbclf_best.predict_proba(X_test)

    return y_pred_train, y_pred_test
#


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()

    print('XGBoost process:')
    y_pred_tr, y_pred_ave = xgb_gridsearch(X_train, X_test, y_train, y_test)
    y_pred_ave = np.argmax(y_pred_ave, axis=1)

    # Evaluation the result
    accu = accuracy_score(y_test, y_pred_ave)
    print('\nAveraged model:')
    print('accuracy = {:>.4f}'.format(accu))

    confmat = confusion_matrix(y_test, y_pred_ave)
    print('\nconfusion matrix:')
    print(confmat)
