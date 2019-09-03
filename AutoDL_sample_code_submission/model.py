# -*- coding: utf-8 -*-

import pandas as pd
import os
import argparse
import time
import jieba
import pickle
import re
import tensorflow as tf
import numpy as np
import sys, getopt
from subprocess import check_output
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from tensorflow.python.keras.preprocessing import text
from tensorflow.python.keras.preprocessing import sequence
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2,SelectPercentile
from sklearn.calibration import CalibratedClassifierCV
MAX_VOCAB_SIZE = 30000
from sklearn.feature_selection import SelectKBest, chi2,SelectPercentile
from sklearn.model_selection import train_test_split


# code form https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
def clean_en_text(dat):
    
    REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')
    
    ret = []
    for line in dat:
        # text = text.lower() # lowercase text
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = BAD_SYMBOLS_RE.sub('', line)
        line = line.strip()
        ret.append(line)
    return ret

def clean_zh_text(dat):
    REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')
    
    ret = []
    for line in dat:
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = line.strip()
        ret.append(line)
    return ret

def feature_select(x,y):
    ch2 = SelectPercentile(chi2, 90)
    ch2.fit(x,y)
    train_x = ch2.transform(x)
    return  train_x,ch2


def _tokenize_chinese_words(text):
    text = text.replace(" ", "")
    return ' '.join(list(text))

def vectorize_data_zh(x_train, x_val=None):
    vectorizer = TfidfVectorizer(ngram_range=(1,1), max_features = MAX_VOCAB_SIZE,analyzer='char')
    if x_val:
        full_text = x_train + x_val
    else:
        full_text = x_train
    # print(full_text)
    vectorizer.fit(full_text)
    train_vectorized = vectorizer.transform(x_train)
    if x_val:
        val_vectorized = vectorizer.transform(x_val)
        return train_vectorized, val_vectorized, vectorizer
    return train_vectorized, vectorizer

def vectorize_data_en(x_train, x_val=None):
    vectorizer = TfidfVectorizer(ngram_range=(1,1), max_features = MAX_VOCAB_SIZE)
    if x_val:
        full_text = x_train + x_val
    else:
        full_text = x_train
    # print(full_text)
    vectorizer.fit(full_text)
    train_vectorized = vectorizer.transform(x_train)
    if x_val:
        val_vectorized = vectorizer.transform(x_val)
        return train_vectorized, val_vectorized, vectorizer
    return train_vectorized, vectorizer

# onhot encode to category
def ohe2cat(label):
    return np.argmax(label, axis=1)


#add feature select
def feature_select(x,y):
    rate = 90
#     if (x.shape[0]<20000):
#         rate = 95
        
    ch2 = SelectPercentile(chi2, rate)
    ch2.fit(x,y)
    train_x = ch2.transform(x)
    return  train_x,ch2

def getdata(x_train,y_train,min_num,allnum):
    train = x_train
    label = y_train.tolist()
    id = list(range(len(label)))
    np.random.shuffle(id)
    id_used = []
    out_train, out_label = [], []
    for class_name in range(y_train.shape[1]):
        num = 0
        while num < min_num:
            for index in id:
                if label[index].index(max(label[index])) == class_name:
                    out_train.append(train[index])
                    out_label.append(label[index])
                    num += 1
                    id_used.append(index)
                if num >= min_num:
                    break

    for index in id:
        if index not in id_used and len(out_label) < allnum:
            out_train.append(train[index])
            out_label.append(label[index])
    return out_train,np.array(out_label)





class Model(object):
    """ model of SVM baseline """

    def __init__(self, metadata):
        """ Initialization for model
        :param metadata: a dict formed like:
            {"class_num": 10,
             "language": ZH,
             "num_train_instances": 10000,
             "num_test_instances": 1000,
             "time_budget": 300}
        """
        self.done_training = False
        self.metadata = metadata
        print("meta data",metadata)
        self.train_count=1
        self.train_output_path = './'
        self.test_input_path = './'
        self.last_num = 0

    def train(self, train_dataset, remaining_time_budget=None):
        """model training on train_dataset.
        
        :param train_dataset: tuple, (x_train, y_train)
            x_train: list of str, input training sentences.
            y_train: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:
        """
        if self.done_training:
            return
        x_train, y_train = train_dataset
        print(type(x_train),type(y_train))
        
        num = len(x_train)
        MAX_COUNT =100
        
#         if num>300000:
#             x_train,y_train = getdata(x_train,y_train,3,300000)
#             num = len(x_train)
            
        
        if (num<=1000):
            MAX_COUNT = 1
            train_num = num
            self.train_count=MAX_COUNT
            
        elif self.train_count==1:
            train_num = 1000
        else:
            train_num = self.last_num*2
            
        if train_num>=num:
            train_num = num
            self.train_count = MAX_COUNT
            
        x_train,y_train = getdata(x_train,y_train,80,train_num)
        y_train = ohe2cat(y_train)
        self.last_num = len(x_train)
        unique, counts = np.unique(y_train, return_counts=True)
        print(np.asarray((unique, counts)).T)
  
        print("这是第%d次训练"%(self.train_count))
        print("train 样本个数:%d"%len(x_train))
        


        # tokenize Chinese words
        if self.metadata['language'] == 'ZH':
            x_train = clean_zh_text(x_train)
            x_train = list(map(_tokenize_chinese_words, x_train))
            x_train, tokenizer = vectorize_data_zh(x_train)
            
            print("特征选择前维度",x_train.shape,y_train.shape)
            x_train, skb = feature_select(x_train,y_train)
            print("特征选择后维度",x_train.shape,y_train.shape)

            with open(self.train_output_path + 'skb.pickle', 'wb') as handle:
                pickle.dump(skb, handle, protocol=pickle.HIGHEST_PROTOCOL)

            
         
            # x_train, y_train = feature_select(x_train, y_train)
            model = LinearSVC(random_state=0, tol=1e-5, max_iter=1000, class_weight='balanced')#
            model = CalibratedClassifierCV(model)
            print(str(type(x_train)) + " " + str(y_train.shape))
            #后面几轮如果出现数据量太大 内存报错 就提前结束
            try:
                model.fit(x_train, y_train)
            except:
                print("训练出错内存出错 提取结束训练")
                return
        else:
            x_train = clean_en_text(x_train)
            x_train, tokenizer = vectorize_data_en(x_train)
            print("特征选择前维度",x_train.shape,y_train.shape)
            x_train, skb = feature_select(x_train,y_train)
            print("特征选择后维度",x_train.shape,y_train.shape)

            with open(self.train_output_path + 'skb.pickle', 'wb') as handle:
                pickle.dump(skb, handle, protocol=pickle.HIGHEST_PROTOCOL)

            
            model = LinearSVC(random_state=0, tol=1e-5, max_iter=1000, class_weight='balanced')
            model = CalibratedClassifierCV(model)
            print(str(type(x_train)) + " " + str(y_train.shape))
            #后面几轮如果出现数据量太大 内存报错 就提前结束
            try:
                model.fit(x_train, y_train)
            except:
                print("训练出错内存出错 提取结束训练")
                return
 

        with open(self.train_output_path + 'model.pickle', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.train_output_path + 'tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if self.train_count>=MAX_COUNT:
            self.done_training=True
            
        self.train_count+=1


    def test(self, x_test, remaining_time_budget=None):
        """
        :param x_test: list of str, input test sentences.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                 here `sample_count` is the number of examples in this dataset as test
                 set and `class_num` is the same as the class_num in metadata. The
                 values should be binary or in the interval [0,1].
        """
        
        with open(self.test_input_path + 'skb.pickle', 'rb') as handle:
            skb = pickle.load(handle, encoding='iso-8859-1')

        with open(self.test_input_path + 'model.pickle', 'rb') as handle:
            model = pickle.load(handle, encoding='iso-8859-1')
        with open(self.test_input_path + 'tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle, encoding='iso-8859-1')

        train_num, test_num = self.metadata['train_num'], self.metadata['test_num']
        class_num = self.metadata['class_num']
        # print('--------------------------------------------------')
        # print(x_test)
        # print(type(x_test))
        # tokenizing Chinese words
        if self.metadata['language'] == 'ZH':
            x_test = clean_zh_text(x_test)
            x_test = list(map(_tokenize_chinese_words, x_test))
        else:
            x_test = clean_en_text(x_test)

        x_test = tokenizer.transform(x_test)
        x_test = skb.transform(x_test)

        # result = model.predict_proba(x_test)#predict
        # print(result)
        #
        # # category class list to sparse class list of lists
        # y_test = np.zeros([test_num, class_num])
        # for idx, y in enumerate(result):
        #     y_test[idx][y] = 1
        # return y_test
        result = model.predict_proba(x_test)
        if result.shape[1] < 2:
            y_test = np.zeros([test_num, class_num])
            y_test[:, 0] = result[:, 0]
            y_test[:, 1] = 1 - result[:, 0]
            result = y_test

        return result
