from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.3.11"
__author__ = "Abien Fred Agarap"


import os
import sys
import tensorflow as tf
import time
from Parser import parameter_parser
from models.loss_draw import LossHistory
import warnings
import numpy as np
import matplotlib.pyplot as plt

acc_gru_svm, loss_gru_svm = [], []
acc_txt_gru_svm = "gru_svm_acc.txt"
acc_txt_read_gru_svm = open(acc_txt_gru_svm, "r")
acc_lines_gru_svm = acc_txt_read_gru_svm.readlines()

for line in acc_lines_gru_svm:
    acc_gru_svm.append(float(line.strip()))

acc_gru_svm = np.array(acc_gru_svm)

loss_txt_gru_svm = "gru_svm_loss.txt"
loss_txt_read_gru_svm = open(loss_txt_gru_svm, "r")
loss_lines_gru_svm = loss_txt_read_gru_svm.readlines()

for line in loss_lines_gru_svm:
    loss_gru_svm.append(float(line.strip()))

loss_gru_svm = np.array(loss_gru_svm)



# ------------------------------gru_softmax------------------------------#
acc_gru_softmax, loss_gru_softmax = [], []
acc_txt_gru_softmax = "gru_softmax_acc6.txt"
acc_txt_read_gru_softmax = open(acc_txt_gru_softmax, "r")
acc_lines_gru_softmax = acc_txt_read_gru_softmax.readlines()

for line in acc_lines_gru_softmax:
    acc_gru_softmax.append(float(line.strip()))

acc_gru_softmax = np.array(acc_gru_softmax)

loss_txt_gru_softmax = "gru_softmax_loss4.txt"
loss_txt_read_gru_softmax = open(loss_txt_gru_softmax, "r")
loss_lines_gru_softmax = loss_txt_read_gru_softmax.readlines()

for line in loss_lines_gru_softmax:
    loss_gru_softmax.append(float(line.strip()))

loss_gru_softmax = np.array(loss_gru_softmax)

plt.figure()
lw = 1
plt.rcParams['figure.figsize'] = (5, 10)


plt.plot(acc_gru_svm, color='red',
         lw=lw, label='BiGAS_acc' , linestyle='-', marker='.',
         markevery=0.05, mew=1.5)

plt.plot(acc_gru_softmax, color='LightSkyBlue',
         lw=lw, label='BiGRU_ATT_Softmax_acc' , linestyle='-', marker='.',
         markevery=0.05, mew=1.5)
plt.rcParams.update({"font.size":19})
plt.tick_params(labelsize=15)
plt.xlim(-0.01, 130.01)
plt.ylim(-0.01, 1.01)
plt.xlabel('step',fontsize=18)
plt.ylabel('accuracy',fontsize=18)
# plt.title('Roc curve comparision between corenodes and fullnodes')
plt.legend(loc="lower right")
plt.savefig("acc.pdf")
plt.show()

# plt.plot(loss_gru_softmax, color='orange',
#           lw=lw, label=' gru_softmax_loss' , linestyle='-', marker='.',
#           markevery=0.05, mew=1.5)

# plt.plot(loss_gru_svm, color='green',
#          lw=lw, label='gru_svm_loss' , linestyle='-', marker='.',
#          markevery=0.05, mew=1.5)
#
# plt.xlim(-0.01, 130.01)
# plt.ylim(-0.01, 1.01)
# plt.xlabel('step')
# plt.ylabel('loss')
# # plt.title('Roc curve comparision between corenodes and fullnodes')
# plt.legend(loc="lower right")
# plt.savefig("Roc15.pdf")
# plt.show()