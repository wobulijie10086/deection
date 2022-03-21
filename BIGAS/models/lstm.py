from __future__ import print_function

import warnings
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, ReLU
from keras.optimizers import Adamax
from sklearn.model_selection import train_test_split
from Parser import parameter_parser
from models.loss_draw import LossHistory

args = parameter_parser()

warnings.filterwarnings("ignore")

"""
LSTM neural network
"""


class LSTM_Model:
    def __init__(self, data, name="", batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, dropout=args.dropout,
                 threshold=args.threshold):
        vectors = np.stack(data.iloc[:, 0].values)
        # vectors = vectors.reshape()
        labels = data.iloc[:, 1].values
        positive_idxs = np.where(labels == 1)[0]
        negative_idxs = np.where(labels == 0)[0]
        undersampled_negative_idxs = np.random.choice(negative_idxs, len(positive_idxs), replace=False)
        resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])

        x_train, x_test, y_train, y_test = train_test_split(vectors[resampled_idxs], labels[resampled_idxs],
                                                            test_size=0.2, stratify=labels[resampled_idxs])
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)
        self.name = name
        self.batch_size = batch_size
        self.epochs = epochs
        self.threshold = threshold
        self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels)
        model = Sequential()
        model.add(LSTM(300, input_shape=(vectors.shape[1], vectors.shape[2])))
        model.add(ReLU())
        model.add(Dropout(dropout))
        model.add(Dense(300))
        model.add(ReLU())
        model.add(Dropout(dropout))
        model.add(Dense(2, activation='softmax'))
        # Lower learning rate to prevent divergence

        adamax = Adamax(lr)
        model.compile(adamax, 'categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    """
    Trains model
    """

    def train(self):
        history = LossHistory()
        self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs,
                       class_weight=self.class_weight, verbose=1, callbacks=[history], validation_data=(self.x_test, self.y_test))
        self.model.save_weights(self.name + "_model.pkl")
        history.loss_plot('epoch')

    """
    Tests accuracy of model
    Loads weights from file if no weights are attached to model object
    """

    def test(self):
        # self.model.load_weights(self.name + "_model.pkl")
        values = self.model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
        print("Accuracy: ", values[1])
        predictions = (self.model.predict(self.x_test, batch_size=self.batch_size)).round()
        # predictions = (predictions >= self.threshold)

        tn, fp, fn, tp = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1)).ravel()
        print("Accuracy: ", (tp + tn) / (tp + tn + fp + fn))
        print('False positive rate(FPR): ', fp / (fp + tn))
        print('False negative rate(FN): ', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('Recall(TPR): ', recall)
        precision = tp / (tp + fp)
        print('Precision: ', precision)
        print('F1 score: ', (2 * precision * recall) / (precision + recall))
