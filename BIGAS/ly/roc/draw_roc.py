import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc  # compute roc and auc
from scipy.interpolate import spline

# Compute ROC curve and ROC area for each class

# ------------------------------gru_svm------------------------------#
fpr_gru_svm, tpr_gru_svm = [], []
fpr_txt_gru_svm = "fpr_gru_svm.txt"
fpr_txt_read_gru_svm = open(fpr_txt_gru_svm, "r")
fpr_lines_gru_svm = fpr_txt_read_gru_svm.readlines()

for line in fpr_lines_gru_svm:
    fpr_gru_svm.append(float(line.strip()))

fpr_gru_svm = np.array(fpr_gru_svm)

tpr_txt_gru_svm = "tpr_gru_svm.txt"
tpr_txt_read_gru_svm = open(tpr_txt_gru_svm, "r")
tpr_lines_gru_svm = tpr_txt_read_gru_svm.readlines()

for line in tpr_lines_gru_svm:
    tpr_gru_svm.append(float(line.strip()))

tpr_gru_svm = np.array(tpr_gru_svm)

roc_auc_gru_svm = auc(fpr_gru_svm, tpr_gru_svm)
print(roc_auc_gru_svm)

# ------------------------------gru_softmax------------------------------#
fpr_gru_softmax, tpr_gru_softmax = [], []
fpr_txt_gru_softmax = "fpr_gru_softmax.txt"
fpr_txt_read_gru_softmax = open(fpr_txt_gru_softmax, "r")
fpr_lines_gru_softmax = fpr_txt_read_gru_softmax.readlines()

for line in fpr_lines_gru_softmax:
    fpr_gru_softmax.append(float(line.strip()))

fpr_gru_softmax = np.array(fpr_gru_softmax)

tpr_txt_gru_softmax = "tpr_gru_softmax.txt"
tpr_txt_read_gru_softmax = open(tpr_txt_gru_softmax, "r")
tpr_lines_gru_softmax = tpr_txt_read_gru_softmax.readlines()

for line in tpr_lines_gru_softmax:
    tpr_gru_softmax.append(float(line.strip()))

tpr_gru_softmax = np.array(tpr_gru_softmax)

roc_auc_gru_softmax = auc(fpr_gru_softmax, tpr_gru_softmax)
print(roc_auc_gru_softmax)


# ------------------------------gru------------------------------#
fpr_gru, tpr_gru = [], []
fpr_txt_gru = "fpr_gru.txt"
fpr_txt_read_gru = open(fpr_txt_gru, "r")
fpr_lines_gru = fpr_txt_read_gru.readlines()

for line in fpr_lines_gru:
    fpr_gru.append(float(line.strip()))

fpr_gru = np.array(fpr_gru)

tpr_txt_gru = "tpr_gru.txt"
tpr_txt_read_gru = open(tpr_txt_gru, "r")
tpr_lines_gru = tpr_txt_read_gru.readlines()

for line in tpr_lines_gru:
    tpr_gru.append(float(line.strip()))

tpr_gru = np.array(tpr_gru)

roc_auc_gru = auc(fpr_gru, tpr_gru)
print(roc_auc_gru)









# ------------------------------rnn------------------------------#
fpr_rnn, tpr_rnn = [], []
fpr_txt_rnn = "fpr_rnn1.txt"
fpr_txt_read_rnn = open(fpr_txt_rnn, "r")
fpr_lines_rnn = fpr_txt_read_rnn.readlines()

for line in fpr_lines_rnn:
    fpr_rnn.append(float(line.strip()))

fpr_rnn = np.array(fpr_rnn)

tpr_txt_rnn = "tpr_rnn1.txt"
tpr_txt_read_rnn = open(tpr_txt_rnn, "r")
tpr_lines_rnn = tpr_txt_read_rnn.readlines()

for line in tpr_lines_rnn:
    tpr_rnn.append(float(line.strip()))

tpr_rnn = np.array(tpr_rnn)

roc_auc_rnn = auc(fpr_rnn, tpr_rnn)
print(roc_auc_rnn)

 #  ------------------------------ lstm ------------------------------ #
fpr_lstm, tpr_lstm = [], []
fpr_txt_lstm = "fpr_lstm1.txt"
fpr_txt_read_lstm = open(fpr_txt_lstm, "r")
fpr_lines_lstm = fpr_txt_read_lstm.readlines()

for line in fpr_lines_lstm:
    fpr_lstm.append(float(line.strip()))

fpr_lstm = np.array(fpr_lstm)
tpr_txt_lstm = "tpr_lstm1.txt"
tpr_txt_read_lstm = open(tpr_txt_lstm, "r")
tpr_lines_lstm = tpr_txt_read_lstm.readlines()

for line in tpr_lines_lstm:
    tpr_lstm.append(float(line.strip()))

tpr_lstm = np.array(tpr_lstm)
roc_auc_lstm = auc(fpr_lstm, tpr_lstm)
print(roc_auc_lstm)


# ------------------------------blstm_att------------------------------#

fpr_blstm_att, tpr_blstm_att = [], []
fpr_txt_blstm_att = "fpr_blstm_att1.txt"
fpr_txt_read_blstm_att = open(fpr_txt_blstm_att, "r")
fpr_lines_blstm_att = fpr_txt_read_blstm_att.readlines()

for line in fpr_lines_blstm_att:
    fpr_blstm_att.append(float(line.strip()))

fpr_blstm_att = np.array(fpr_blstm_att)
tpr_txt_blstm_att = "tpr_blstm_att1.txt"
tpr_txt_read_blstm_att = open(tpr_txt_blstm_att, "r")
tpr_lines_blstm_att = tpr_txt_read_blstm_att.readlines()

for line in tpr_lines_blstm_att:
    tpr_blstm_att.append(float(line.strip()))

tpr_blstm_att = np.array(tpr_blstm_att)
roc_auc_blstm_att = auc(fpr_blstm_att, tpr_blstm_att)
print(roc_auc_blstm_att)


# ------------------------------blstm------------------------------#

fpr_blstm, tpr_blstm = [], []
fpr_txt_blstm = "fpr_blstm1.txt"
fpr_txt_read_blstm = open(fpr_txt_blstm, "r")
fpr_lines_blstm = fpr_txt_read_blstm.readlines()

for line in fpr_lines_blstm:
    fpr_blstm.append(float(line.strip()))

fpr_blstm = np.array(fpr_blstm)
tpr_txt_blstm = "tpr_blstm1.txt"
tpr_txt_read_blstm = open(tpr_txt_blstm, "r")
tpr_lines_blstm = tpr_txt_read_blstm.readlines()

for line in tpr_lines_blstm:
    tpr_blstm.append(float(line.strip()))

tpr_blstm = np.array(tpr_blstm)
roc_auc_blstm = auc(fpr_blstm, tpr_blstm)
print(roc_auc_blstm)





plt.figure()
lw = 1
plt.rcParams['figure.figsize'] = (5, 3.5)

plt.plot(fpr_gru_svm, tpr_gru_svm, color='darkorange',
         lw=lw, label='ROC curve of BiGAS (AUC = %0.4f)' % roc_auc_gru_svm, linestyle='-', marker='.',
         markevery=0.05, mew=1.5)

plt.plot(fpr_gru_softmax, tpr_gru_softmax, color='brown',
         lw=lw, label='ROC curve of BiGRU_ATT_Softmax (AUC = %0.4f)' % roc_auc_gru_softmax, linestyle=':', marker='+',
         markevery=0.05, mew=1.5)

plt.plot(fpr_blstm_att, tpr_blstm_att, color='black',
         lw=lw, label='ROC curve of BLSTM_att (AUC = %0.4f)' % roc_auc_blstm_att, linestyle='--', marker='+',
         markevery=0.05, mew=1.5)

plt.plot(fpr_blstm, tpr_blstm, color='blue',
         lw=lw, label='ROC curve of BLSTM (AUC = %0.4f)' % roc_auc_blstm, linestyle='-.', marker='*',
         markevery=0.05, mew=1.5)
plt.plot(fpr_gru, tpr_gru, color='green',
         lw=lw, label='ROC curve of GRU (AUC = %0.4f)' % roc_auc_gru, linestyle=':', marker='.',
         markevery=0.05, mew=1.5)
plt.plot(fpr_lstm, tpr_lstm, color='red',
         lw=lw, label='ROC curve of LSTM (AUC = %0.4f)' % roc_auc_lstm, linestyle='-', marker='x',
         markevery=0.05, mew=1.5)

plt.plot(fpr_rnn, tpr_rnn, color='purple',
         lw=lw, label='ROC curve of rnn (AUC = %0.4f)' % roc_auc_rnn, linestyle='-', marker='.',
         markevery=0.05, mew=1.5)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Roc curve comparision between corenodes and fullnodes')
plt.legend(loc="lower right")
plt.savefig("Roc21.pdf")
plt.show()
