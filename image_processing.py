import argparse
import os
import shutil

import splitfolders

from os.path import exists, join
from PIL import Image


def random_splitter(src, dest, test_rate, create_grayscale):
    """Splits generated images (slices) into train/val/test subdirectories.

        Parameters
        ----------
        src : str
            Path to the slices directory.

        dest : str
            Path to the destination directory.

        test_rate : float
            Test images rate.

        create_grayscale : boolean
            Enables or disables grayscale targets creation.
        """

    ds_path = join(src, "dataset/")
    full_ds = join(src, "full/")

    inputs = join(full_ds, "rgb")
    targets = join(full_ds, "grayscale")

    if exists(ds_path) is False:
        os.makedirs(ds_path)
    if exists(full_ds) is False:
        os.makedirs(full_ds)
    if exists(inputs) is False:
        os.makedirs(inputs)
    if exists(targets) is False:
        os.makedirs(targets)

    if create_grayscale:
        k = 0

        for root, _, files in os.walk(src):
            for f in files:
                if not f.endswith(".DS_Store"):
                    file = join(root, f)
                    filename = f.split('.')[0]
                    img_format = f.split(".")[-1]
                    rgb_img = Image.open(file)
                    lab_img = rgb_img.convert('L')

                    if exists(join(inputs, f"{filename}.{img_format}")) is False:
                        rgb_img.save(join(inputs, f"{filename}.{img_format}"))
                    if exists(join(targets, f"{filename}.{img_format}")) is False:
                        lab_img.save(join(targets, f"{filename}.{img_format}"))
                    if exists(join(ds_path, f)) is False:
                        k += 1
                        shutil.move(file, ds_path)

        print(f"{k} images found")

    assert test_rate < 1, "test_rate must be less than 1"

    val_rate = 0.1

    splitfolders.ratio(full_ds, output=dest,
                       seed=1337, ratio=(1 - (test_rate + val_rate), val_rate, test_rate), group_prefix=None,
                       move=False)

    for root, directories, files in os.walk(full_ds):
        for d in directories:
            if d == "train" or d == "val" or d == "test":
                splitted = join(src, "splitted/")
                if exists(splitted) is False:
                    os.makedirs(splitted)
                shutil.move(join(full_ds, d), splitted)

    print("Splitting done")


def main():
    """This is the main function"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="path to the dataset")
    parser.add_argument("--output", "-o", help="path to the output")
    parser.add_argument("--testrate", "-t", help="part of test set", default=0.2)
    parser.add_argument("--create_grayscale", "-c", help="clear useless images", default=False, action="store_true")

    args = parser.parse_args()

    datapath = args.datapath
    output = args.output
    test_rate = args.testrate
    create_grayscale = args.create_grayscale

    random_splitter(datapath, output, test_rate, create_grayscale)


if __name__ == "__main__":
    main()