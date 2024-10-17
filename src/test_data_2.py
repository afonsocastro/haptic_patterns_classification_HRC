#!/usr/bin/env python3

import matplotlib.pyplot as plt
import keras
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import numpy as np
from utils import plot_confusion_matrix_percentage

if __name__ == '__main__':
    # model_option = "cnn"
    # model_option = "lstm"
    model_option = "transformer"

    x_test_data = np.load("test_data/data_1.npy")
    y_test_data = np.load("test_data/ground_truth_data_1.npy")

    if model_option == "cnn":
        cnn_model = keras.models.load_model("convolutional_model")
        predicted_values = cnn_model.predict(x=x_test_data, verbose=2)
        # Reverse to_categorical from keras utils
        predicted_values = np.argmax(predicted_values, axis=1, out=None)

        cm = confusion_matrix(y_true=y_test_data, y_pred=predicted_values)

        labels = ['PULL', 'PUSH', 'SHAKE', 'TWIST']
        n_labels = len(labels)
        blues = plt.cm.Blues
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=blues)

        plt.show()
        # plt.savefig(ROOT_DIR + "/neural_networks/convolutional/predicted_data/confusion_matrix.png", bbox_inches='tight')

        cm_true = cm / cm.astype(float).sum(axis=1)
        cm_true_percentage = cm_true * 100
        plot_confusion_matrix_percentage(confusion_matrix=cm_true_percentage, display_labels=labels, cmap=blues,
                                         title="Confusion Matrix (%) - CONVOLUTIONAL")
        plt.show()

    elif model_option == "lstm":
        lstm_model = keras.models.load_model("recurrent_model")

    elif model_option == "transformer":
        encoder_model = keras.models.load_model("transformer_model")


