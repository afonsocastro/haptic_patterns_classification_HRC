#!/usr/bin/env python3

import matplotlib.pyplot as plt
import keras
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import numpy as np
from PDF import PDF
from utils import plot_confusion_matrix_percentage, prediction_classification_absolute, simple_metrics_calc
# from neural_networks.utils import plot_confusion_matrix_percentage, prediction_classification, simple_metrics_calc, \
#     prediction_classification_absolute

if __name__ == '__main__':
    model_option = "cnn"
    # model_option = "lstm"
    # model_option = "transformer"

    x_test_data = np.load("../test_data/data_1.npy")
    y_test_data = np.load("../test_data/ground_truth_data_1.npy")

    predictions_list = []

    pull = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    push = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    shake = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    twist = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}

    metrics_pull = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}
    metrics_push = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}
    metrics_shake = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}
    metrics_twist = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}

    if model_option == "cnn":
        cnn_model = keras.models.load_model("../models/convolutional_model")

        for i in range(0, len(x_test_data)):
            prediction = cnn_model.predict(x=x_test_data[i:i+1, :, :, :], verbose=2)
            # Reverse to_categorical from keras utils
            decoded_prediction = np.argmax(prediction, axis=1, out=None)
            true = int(y_test_data[i])
            prediction_classification_absolute(cla=0, true_out=true, dec_pred=decoded_prediction, dictionary=pull)
            prediction_classification_absolute(cla=1, true_out=true, dec_pred=decoded_prediction, dictionary=push)
            prediction_classification_absolute(cla=2, true_out=true, dec_pred=decoded_prediction, dictionary=shake)
            prediction_classification_absolute(cla=3, true_out=true, dec_pred=decoded_prediction, dictionary=twist)

            predictions_list.append(decoded_prediction)

        predicted_values = np.asarray(predictions_list)

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

        # METRICS-----------------------------------------------------------------------------------------
        simple_metrics_calc(pull, metrics_pull)
        simple_metrics_calc(push, metrics_push)
        simple_metrics_calc(shake, metrics_shake)
        simple_metrics_calc(twist, metrics_twist)

        metrics = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}

        for m in ["accuracy", "recall", "precision", "f1"]:
            metrics[m] = (metrics_pull[m] + metrics_push[m] + metrics_shake[m] + metrics_twist[m]) / 4

        data = [
            ["", "Accuracy", "Precision", "Recall", "F1", ],
            ["", str(round(metrics["accuracy"], 4)), str(round(metrics["precision"], 4)),
             str(round(metrics["recall"], 4)),
             str(round(metrics["f1"], 4)), ]
        ]

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Times", size=10)
        pdf.create_table(table_data=data, title='CONVOLUTIONAL Metrics Dataset 1', cell_width='uneven', x_start=25)
        pdf.ln()
        pdf.output('../metrics/test_dataset_1/convolutional_metrics_data_1.pdf')

    elif model_option == "lstm":
        lstm_model = keras.models.load_model("../models/recurrent_model")
        for i in range(0, len(x_test_data)):
            prediction = lstm_model.predict(x=x_test_data[i:i+1, :, :, :], verbose=2)
            # Reverse to_categorical from keras utils
            decoded_prediction = np.argmax(prediction, axis=1, out=None)
            true = int(y_test_data[i])
            prediction_classification_absolute(cla=0, true_out=true, dec_pred=decoded_prediction, dictionary=pull)
            prediction_classification_absolute(cla=1, true_out=true, dec_pred=decoded_prediction, dictionary=push)
            prediction_classification_absolute(cla=2, true_out=true, dec_pred=decoded_prediction, dictionary=shake)
            prediction_classification_absolute(cla=3, true_out=true, dec_pred=decoded_prediction, dictionary=twist)

            predictions_list.append(decoded_prediction)

        predicted_values = np.asarray(predictions_list)

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
                                         title="Confusion Matrix (%) - RECURRENT")
        plt.show()

        # METRICS-----------------------------------------------------------------------------------------
        simple_metrics_calc(pull, metrics_pull)
        simple_metrics_calc(push, metrics_push)
        simple_metrics_calc(shake, metrics_shake)
        simple_metrics_calc(twist, metrics_twist)

        metrics = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}

        for m in ["accuracy", "recall", "precision", "f1"]:
            metrics[m] = (metrics_pull[m] + metrics_push[m] + metrics_shake[m] + metrics_twist[m]) / 4

        data = [
            ["", "Accuracy", "Precision", "Recall", "F1", ],
            ["", str(round(metrics["accuracy"], 4)), str(round(metrics["precision"], 4)),
             str(round(metrics["recall"], 4)),
             str(round(metrics["f1"], 4)), ]
        ]

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Times", size=10)
        pdf.create_table(table_data=data, title='RECURRENT Metrics Dataset 1', cell_width='uneven', x_start=25)
        pdf.ln()
        pdf.output('../metrics/test_dataset_1/recurrent_metrics_data_1.pdf')

    elif model_option == "transformer":
        encoder_model = keras.models.load_model("../models/transformer_model")

        for i in range(0, len(x_test_data)):
            prediction = encoder_model.predict(x=x_test_data[i:i + 1, :, :, :], verbose=2)
            # Reverse to_categorical from keras utils
            decoded_prediction = np.argmax(prediction, axis=1, out=None)
            true = int(y_test_data[i])
            prediction_classification_absolute(cla=0, true_out=true, dec_pred=decoded_prediction, dictionary=pull)
            prediction_classification_absolute(cla=1, true_out=true, dec_pred=decoded_prediction, dictionary=push)
            prediction_classification_absolute(cla=2, true_out=true, dec_pred=decoded_prediction, dictionary=shake)
            prediction_classification_absolute(cla=3, true_out=true, dec_pred=decoded_prediction, dictionary=twist)

            predictions_list.append(decoded_prediction)

        predicted_values = np.asarray(predictions_list)

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
                                         title="Confusion Matrix (%) - TRANSFORMER")
        plt.show()
        # METRICS-----------------------------------------------------------------------------------------
        simple_metrics_calc(pull, metrics_pull)
        simple_metrics_calc(push, metrics_push)
        simple_metrics_calc(shake, metrics_shake)
        simple_metrics_calc(twist, metrics_twist)

        metrics = {"accuracy": 0, "recall": 0, "precision": 0, "f1": 0}

        for m in ["accuracy", "recall", "precision", "f1"]:
            metrics[m] = (metrics_pull[m] + metrics_push[m] + metrics_shake[m] + metrics_twist[m]) / 4

        data = [
            ["", "Accuracy", "Precision", "Recall", "F1", ],
            ["", str(round(metrics["accuracy"], 4)), str(round(metrics["precision"], 4)),
             str(round(metrics["recall"], 4)),
             str(round(metrics["f1"], 4)), ]
        ]

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Times", size=10)
        pdf.create_table(table_data=data, title='TRANSFORMER Metrics Dataset 1', cell_width='uneven', x_start=25)
        pdf.ln()
        pdf.output('../metrics/test_dataset_1/transformer_metrics_data_1.pdf')



