from tensorflow.keras.utils import to_categorical  # one-hot encode target column
import matplotlib.pyplot as plt
import keras
import numpy as np


if __name__ == '__main__':
    cnn_model = keras.models.load_model("convolutional_model")
    lstm_model = keras.models.load_model("recurrent_model")
    encoder_model = keras.models.load_model("transformer_model")
    print(cnn_model .summary())
    print(lstm_model .summary())
    print(encoder_model .summary())
    keras.utils.plot_model(cnn_model, to_file='../models/cnn_model_plot.png', show_shapes=True, show_layer_names=True)
    keras.utils.plot_model(lstm_model, to_file='../models/lstm_model_plot.png', show_shapes=True, show_layer_names=True)
    keras.utils.plot_model(encoder_model, to_file='../models/encoder_model_plot.png', show_shapes=True, show_layer_names=True)
