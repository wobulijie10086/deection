# 开发时间：2021/6/11 14:44
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.5.4"
__author__ = "Abien Fred Agarap"


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf

import warnings
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, GRU
from keras.optimizers import Adamax
from sklearn.model_selection import train_test_split
from Parser import parameter_parser
from models.loss_draw import LossHistory


args = parameter_parser()

warnings.filterwarnings("ignore")
def load_data(
         data, name="", batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, dropout=args.dropout,
):



         #print(data)

         vectors = np.stack(data.iloc[:, 0].values)

         # print(vectors)
         #print(vectors.shape)
         labels = data.iloc[:, 1].values

        # print(labels)
        # print(labels.shape)
         positive_idxs = np.where(labels == 1)[0]
        # print(positive_idxs)
        # print(positive_idxs.shape)
         negative_idxs = np.where(labels == 0)[0]
        # print(negative_idxs)
        # print(negative_idxs.shape)
         idxs = np.concatenate([positive_idxs, negative_idxs])
        # print(idxs)
        # print(idxs.shape)
         undersampled_negative_idxs = np.random.choice(negative_idxs, len(positive_idxs), replace=False)
        # print(undersampled_negative_idxs.shape)
         resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])
        # print(resampled_idxs.shape)
        # print(resampled_idxs)
         x_train, x_test, y_train, y_test = train_test_split(vectors[resampled_idxs], labels[resampled_idxs],
                                                            test_size=0.2, stratify=labels[ resampled_idxs])
         x_train = x_train
         x_test = x_test
         y_train=y_train
         y_test=y_test
         #y_train = to_categorical(y_train)
         #y_test = to_categorical(y_test)

         batch_size = args.batch_size
         epochs = args.epochs
         class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels)

         # print(x_train.shape)
         # print(y_train.shape)
         # print(y_train)

         return x_train, y_train, x_test, y_test


def load_data2(data):
        vectors = np.stack(data.iloc[:, 0].values)

        # print(vectors)
        # print(vectors.shape)
        labels = data.iloc[:, 1].values

        # print(labels)
        # print(labels.shape)
        positive_idxs = np.where(labels == 1)[0]
        # print(positive_idxs)
        # print(positive_idxs.shape)
        negative_idxs = np.where(labels == 0)[0]
        # print(negative_idxs)
        # print(negative_idxs.shape)
        idxs = np.concatenate([positive_idxs, negative_idxs])
        # print(idxs)
        # print(idxs.shape)
        undersampled_negative_idxs = np.random.choice(negative_idxs, len(positive_idxs), replace=False)
        # print(undersampled_negative_idxs.shape)
        resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])
        # print(resampled_idxs.shape)
        # print(resampled_idxs)
        x_train, x_test, y_train, y_test = train_test_split(vectors[resampled_idxs], labels[resampled_idxs],
                                                            test_size=0.2, stratify=labels[resampled_idxs])
        x_train = x_train
        x_test = x_test
        # y_train = to_categorical(y_train)
        # y_test = to_categorical(y_test)
        y_train = y_train
        y_test = y_test
        batch_size = args.batch_size
        epochs = args.epochs
        class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels)

        # print(x_train.shape)
        # print(y_train.shape)
        # print(y_train)

        return x_test, y_test