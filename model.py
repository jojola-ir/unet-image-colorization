'''custom model building'''

import argparse
import os
from datetime import datetime

from tensorflow import keras


def unet(n_levels, initial_features=64, n_blocks=2, kernel_size=3,
                    pooling_size=2, in_channels=1, out_channels=1):
    """Build a neural network composed of UNET architecture.

    Parameters
    ----------
    clNbr : int
        Number of classes that defines output

    da: boolean
        Enables or disable data augmentation

    Returns
    -------
    model
        builded model
    """
    IMAGE_HEIGHT = 160
    IMAGE_WIDTH = 160

    inputs = keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
    x = inputs

    convpars = dict(kernel_size=kernel_size, activation="relu", padding="same")

    # downstream
    skips = {}
    for level in range(n_levels):
        for block in range(n_blocks):
            x = keras.layers.SeparableConv2D(initial_features * 2 ** level, **convpars)(x)
            if level <= n_levels // 2 and block == 0:
                x = keras.layers.BatchNormalization()(x)
            elif level > n_levels // 2 and level < n_levels - 1 and block == 0:
                x = keras.layers.BatchNormalization()(x)
        if level < n_levels - 1:
            skips[level] = x
            x = keras.layers.MaxPool2D(pooling_size, padding="same")(x)

    # upstream
    for level in reversed(range(n_levels - 1)):
        x = keras.layers.Conv2DTranspose(initial_features * 2 ** level, strides=pooling_size, **convpars)(x)
        x = keras.layers.Concatenate()([x, skips[level]])
        for block in range(n_blocks):
            x = keras.layers.SeparableConv2D(initial_features * 2 ** level, **convpars)(x)

    activation = "sigmoid" if out_channels == 1 else "softmax"
    x = keras.layers.SeparableConv2D(out_channels, kernel_size=1, activation=activation, padding="same")(x)
    model = keras.Model(inputs=[inputs], outputs=x, name=f"UNET-L{n_levels}-F{initial_features}")

    return model



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--load", default=False,
                        help="load previous model",
                        action="store_true")
    parser.add_argument("--modelpath", default="/models/run1.h5",
                        help="path to .h5 file for transfert learning")

    args = parser.parse_args()

    model = unet(5)

    model.summary()

    model_architecture_path = "architecture/test/"
    if os.path.exists(model_architecture_path) is False:
        os.makedirs(model_architecture_path)

    now = datetime.now()
    keras.utils.plot_model(model,
                           to_file=os.path.join(model_architecture_path,
                                                f"model_unet_{now.strftime('%m_%d_%H_%M')}.png"),
                           show_shapes=True)