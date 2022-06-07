'''Dataloader'''

import os
from glob import glob

from os.path import join

import numpy as np
import tensorflow as tf

from skimage.color import rgb2lab


@tf.function
def parse_image(path):
    """Load an RGB image and its grayscale version and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an RGB image and its grayscale version.
    """

    a_path = tf.strings.regex_replace(path, "l_chan", "a_chan")
    b_path = tf.strings.regex_replace(path, "l_chan", "b_chan")

    L = tf.io.read_file(path)
    L = tf.image.decode_jpeg(L, channels=1)
    L = tf.image.convert_image_dtype(L, tf.float32)

    a = tf.io.read_file(a_path)
    a = tf.image.decode_jpeg(a, channels=1)
    a = tf.image.convert_image_dtype(a, tf.float32)

    b = tf.io.read_file(b_path)
    b = tf.image.decode_jpeg(b, channels=1)
    b = tf.image.convert_image_dtype(b, tf.float32)

    ab = tf.experimental.numpy.dstack((a,b))

    return {"L": L, "ab": ab}


@tf.function
def normalize(input_image, input_target):
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of size [SIZE,SIZE,3].
    input_target : tf.Tensor
        Tensorflow tensor containing a grayscale image of size [SIZE,SIZE,1].

    Returns
    -------
    tuple
        Normalized image and its annotation.
    """
    input_image = tf.cast(input_image, tf.float32) / 100
    input_target = tf.cast(input_target, tf.float32) / 128

    return input_image, input_target


@tf.function
def load_image_train(datapoint):
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """

    IMG_SIZE = 160

    input_image = tf.image.resize_with_pad(datapoint["L"], IMG_SIZE, IMG_SIZE)
    input_target = tf.image.resize_with_pad(datapoint["ab"], IMG_SIZE, IMG_SIZE)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_target = tf.image.flip_left_right(input_target)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_up_down(input_image)
        input_target = tf.image.flip_up_down(input_target)

    if tf.random.uniform(()) > 0.5:
        rd = np.random.uniform(low=0.4, high=1.0)
        input_image = tf.image.central_crop(input_image, central_fraction=rd)
        input_target = tf.image.central_crop(input_target, central_fraction=rd)

        input_image = tf.image.resize_with_pad(input_image, IMG_SIZE, IMG_SIZE)
        input_target = tf.image.resize_with_pad(input_target, IMG_SIZE, IMG_SIZE)

    input_image, input_target = normalize(input_image, input_target)

    return input_image, input_target


@tf.function
def load_image_test(datapoint):
    """Normalize and resize a test image and its annotation.

    Notes
    -----
    Since this is for the test set, we don't need to apply
    any data augmentation technique.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """

    IMG_SIZE = 160

    input_image = tf.image.resize_with_pad(datapoint["L"], IMG_SIZE, IMG_SIZE)
    input_target = tf.image.resize_with_pad(datapoint["ab"], IMG_SIZE, IMG_SIZE)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_target = tf.image.flip_left_right(input_target)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_up_down(input_image)
        input_target = tf.image.flip_up_down(input_target)

    input_image, input_target = normalize(input_image, input_target)

    return input_image, input_target


def create_pipeline_performance(path, bs=32):
    """Creates datasets from a directory given as parameter.

    The set given as input must include training and validation directory.

    Parameters
    ----------
    path_to_dataset : str
        Path to the dataset directory.

    bs : int
        Batch size

    Returns
    -------
    train_dataset, val_dataset, test_dataset
        datasets for training, validating and testing
    """

    TRAIN_SEED = 202
    VAL_SEED = 505
    TEST_SEED = 909

    BUFFER_SIZE = 1000

    train_dir = os.path.join(path, "train/")
    val_dir = os.path.join(path, "val/")
    test_dir = os.path.join(path, "test/")

    data_path = join(path, "lab_dataset.npz")

    with np.load(data_path) as data:
        train = {"L": data["X_train"], "ab": data["y_train"]}
        val = {"L": data["X_val"], "ab": data["y_val"]}
        test = {"L": data["X_test"], "ab": data["y_test"]}

    train_dataset = tf.data.Dataset.from_tensor_slices((train))
    val_dataset = tf.data.Dataset.from_tensor_slices((val))
    test_dataset = tf.data.Dataset.from_tensor_slices((test))

    train_num = len(train["L"])
    val_num = len(val["L"])
    test_num = len(test["L"])

    dataset = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

    # -- Train Dataset --#
    dataset["train"] = dataset["train"].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    dataset["train"] = dataset["train"].cache()
    dataset["train"] = dataset["train"].shuffle(buffer_size=BUFFER_SIZE)
    dataset["train"] = dataset["train"].repeat()
    dataset["train"] = dataset["train"].batch(bs)
    dataset["train"] = dataset["train"].prefetch(buffer_size=tf.data.AUTOTUNE)

    # -- Validation Dataset --#
    dataset["val"] = dataset["val"].map(load_image_test)
    dataset["val"] = dataset["val"].repeat()
    dataset["val"] = dataset["val"].batch(bs)
    dataset["val"] = dataset["val"].prefetch(buffer_size=tf.data.AUTOTUNE)

    # -- Test Dataset --#
    dataset["test"] = dataset["test"].map(load_image_test)
    dataset["test"] = dataset["test"].repeat()
    dataset["test"] = dataset["test"].batch(bs)
    dataset["test"] = dataset["test"].prefetch(buffer_size=tf.data.AUTOTUNE)

    print(f"{train_num} images found in {train_dir}.")
    print(f"{val_num} images found in {val_dir}.")
    print(f"{test_num} images found in {test_dir}.")

    return dataset["train"], dataset["val"], dataset["test"], train_num, test_num