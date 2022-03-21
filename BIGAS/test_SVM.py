import numpy as np
from sklearn.metrics import confusion_matrix
test0 = np.load("D:/代码/实验代码/新LSTM：ReChecker-master/ly/results/gru_svm2/testing-gru_svm-0.npy")
test1 = np.load("D:/代码/实验代码/新LSTM：ReChecker-master/ly/results/gru_svm2/testing-gru_svm-1.npy")
test2 = np.load("D:/代码/实验代码/新LSTM：ReChecker-master/ly/results/gru_svm2/testing-gru_svm-2.npy")
test3 = np.load("D:/代码/实验代码/新LSTM：ReChecker-master/ly/results/gru_svm1/testing-gru_svm-3.npy")
test4 = np.load("D:/代码/实验代码/新LSTM：ReChecker-master/ly/results/gru_svm1/testing-gru_svm-4.npy")
test5 = np.load("D:/代码/实验代码/新LSTM：ReChecker-master/ly/results/gru_svm1/testing-gru_svm-5.npy")



# c = np.row_stack((test0,test1,test2,test3,test4,test5))
c = np.row_stack((test0,test1,test2))
# print(c.shape)
a=np.hsplit(c, 2)[0]
b=np.hsplit(c, 2)[1]
# print(a.shape)
# print(b.shape)

tn, fn, fp, tp = confusion_matrix(np.argmax(a, axis=1), np.argmax(b, axis=1)).ravel()

# print(tn, fn, fp, tp)

print('Accuracy:', (tn + tp) / (tn + fp + fn + tp))
print('False positive rate(FPR): ', fp / (fp + tn))
print('False negative rate(FNR): ', fn / (fn + tp))
recall = tp / (tp + fn)
print('Recall(TPR): ', recall)
precision = tp / (tp + fp)
print('Precision: ', precision)
print('F1 score: ', (2 * precision * recall) / (precision + recall))