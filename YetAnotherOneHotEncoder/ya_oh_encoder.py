# -*- coding: utf-8 -*-
#
#   ya_oh_encoder.py
#       date. 9/23/2016
#

import numpy as np

# scikit-learn library is loaded for benchmark
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class YaLabelEncoder():
    '''
      yet another label encoder:
        encode catecorical data by mapping dummy numbers
      usage:
        > yle = YaLabelEncoder()
        > yle.fit(data)
        > encoded = yle.transform(data)
    '''
    def __init__(self):
        self.dic = {}   # dictionary for mapping

    def fit(self, data):
        assert type(data) == np.ndarray
        u = np.unique(data)
        idx = range(len(u))
        d = dict(zip(u, idx))
        self.dic = d

    def transform(self, data):
        orig_shape = data.shape
        d = self.dic
        if len(d) == 0:
            raise ValueError('Encoder instance may not be fitted.')
        encoded = [d[e] for e in data]
        encoded = np.asarray(encoded).reshape(orig_shape)

        return encoded
#

class YaOneHotEncoder():
    '''
      yet another one-hot encoder:
        encode categorical data to one-hot data matrix
        This can substitute sklearn one-hot encoder
      usage:
        > yohe = YaOneHotEncoder()
        > yohe.fit(data)
        > encoded = yohe.transform(data)
    '''
    def __init__(self):
        self.unq = []   # sorted unique values
        self.ulen = 0   # length of unique value list
    
    def fit(self, data):
        assert type(data) == np.ndarray
        u = np.unique(data)
        self.unq = u
        self.ulen = len(u)

    def transform(self, data):
        data = data.reshape((-1, 1))
        transformed = []
        ulen = self.ulen
        if ulen == 0:
            raise ValueError('Encoder instance may not be fitted.')
        for d in data:
            oneline = [0. for i in range(ulen)]
            col_num = (list(self.unq)).index(d)
            oneline[col_num] = 1.
            transformed.append(oneline)
        
        transformed = np.asarray(transformed)
        
        return transformed
#

def test_yle(data):
    yle = YaLabelEncoder()
    yle.fit(data)
    encoded = yle.transform(data)

    return encoded

def test_yohe(data):
    yohe = YaOneHotEncoder()
    yohe.fit(data)
    encoded = yohe.transform(data)

    return encoded

def sample_data():
    base_data = ['P', 'Y', 'T', 'H', 'O', 'N']
    data = np.random.choice(base_data, size=30)

    return data

def one_hot_encoder_by_sklearn(data):
    # Label Encoding
    le = LabelEncoder()
    data_lencoded = le.fit_transform(data)

    # One-hot Encoding
    data_lencoded = data_lencoded.reshape((-1, 1))
    ohe = OneHotEncoder(categorical_features='all')
    data_ohencoded = ohe.fit_transform(data_lencoded).toarray()

    return data_ohencoded

data = sample_data()
encoded1 = one_hot_encoder_by_sklearn(data)
encoded2 = test_yohe(data) 

# check result
print('=====\n', data)          # source
print('=====\n', encoded1[:10]) # by scikit-learn
print('=====\n', encoded2[:10]) # by my module
