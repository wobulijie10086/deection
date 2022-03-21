import numpy as np
import matplotlib.pyplot as plt
# from sklearn.metrics import auc

# ------------------------------gru_svm------------------------------#
recall_gru_svm, precision_gru_svm = [], []
recall_txt_gru_svm = "recall_gru_svm.txt"
recall_txt_read_gru_svm = open(recall_txt_gru_svm, "r")
recall_lines_gru_svm = recall_txt_read_gru_svm.readlines()

for line in recall_lines_gru_svm:
    recall_gru_svm.append(float(line.strip()))

recall_gru_svm = np.array(recall_gru_svm)

precision_txt_gru_svm = "precision_gru_svm.txt"
precision_txt_read_gru_svm = open(precision_txt_gru_svm, "r")
precision_lines_gru_svm = precision_txt_read_gru_svm.readlines()

for line in precision_lines_gru_svm:
    precision_gru_svm.append(float(line.strip()))

precision_gru_svm = np.array(precision_gru_svm)

# roc_auc_gru_svm = auc(recall_gru_svm, precision_gru_svm)
# print(roc_auc_gru_svm)

# ------------------------------gru_softmax------------------------------#
recall_gru_softmax, precision_gru_softmax = [], []
recall_txt_gru_softmax = "recall_gru_softmax.txt"
recall_txt_read_gru_softmax = open(recall_txt_gru_softmax, "r")
recall_lines_gru_softmax = recall_txt_read_gru_softmax.readlines()

for line in recall_lines_gru_softmax:
    recall_gru_softmax.append(float(line.strip()))

recall_gru_softmax = np.array(recall_gru_softmax)

precision_txt_gru_softmax = "precision_gru_softmax.txt"
precision_txt_read_gru_softmax = open(precision_txt_gru_softmax, "r")
precision_lines_gru_softmax = precision_txt_read_gru_softmax.readlines()

for line in precision_lines_gru_softmax:
    precision_gru_softmax.append(float(line.strip()))

precision_gru_softmax = np.array(precision_gru_softmax)

# roc_auc_gru_softmax = auc(recall_gru_softmax, precision_gru_softmax)
# print(roc_auc_gru_softmax)

plt.figure()
lw = 2
plt.rcParams['figure.figsize'] = (5, 3.5)

plt.plot(recall_gru_svm,precision_gru_svm , color='darkorange',
         lw=lw, label='PRC curve of gru_svm ' , linestyle='-', marker='.',
         markevery=0.05, mew=1.5)

plt.plot(recall_gru_softmax,precision_gru_softmax, color='brown',
         lw=lw, label='PRC curve of gru_softmax ' , linestyle='-', marker='.',
         markevery=0.05, mew=1.5)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)
plt.xlabel('Recall')
plt.ylabel('Precision')
# plt.title('Roc curve comparision between corenodes and fullnodes')
plt.legend(loc="lower right")
plt.savefig("Roc8.pdf")
plt.show()