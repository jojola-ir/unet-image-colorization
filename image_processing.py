import numpy as np
import argparse
import os
import shutil

import splitfolders
from tqdm import tqdm

from os.path import exists, join
from skimage import io
from skimage.color import rgb2lab
from sklearn.model_selection import train_test_split
from PIL import Image, ImageCms


def raw_splitter(src, dest, test_rate, create_lab):
    """Splits generated images (slices) into train/val/test subdirectories.

        Parameters
        ----------
        src : str
            Path to the slices directory.

        dest : str
            Path to the destination directory.

        test_rate : float
            Test images rate.

        create_lab : boolean
            Enables or disables grayscale targets creation.
        """

    ds_path = join(src, "dataset/")
    full_ds = join(src, "full/")

    #lab = join(full_ds, "lab")

    l = join(full_ds, "l_chan/")
    a = join(full_ds, "a_chan/")
    b = join(full_ds, "b_chan/")

    if exists(ds_path) is False:
        os.makedirs(ds_path)
    if exists(full_ds) is False:
        os.makedirs(full_ds)
    if exists(l) is False:
        os.makedirs(l)
    if exists(a) is False:
        os.makedirs(a)
    if exists(b) is False:
        os.makedirs(b)
    #if exists(lab) is False:
        #os.makedirs(lab)


    if create_lab:
        k = 0
        file_count = len([name for name in os.listdir(src) if name.endswith(".jpg")])
        print("Converting RGB images to LAB...")

        for root, _, files in os.walk(src):
            for f in tqdm(files, total=file_count, initial=k):
                if not f.endswith(".DS_Store"):
                    file = join(root, f)
                    filename = f.split('.')[0]
                    img_format = f.split(".")[-1]
                    #rgb_img = io.imread(file)
                    #lab_img = rgb2lab(rgb_img)
                    #lab_img = img_as_ubyte(lab_img)
                    rgb_img = Image.open(file).convert("RGB")
                    srgb_p = ImageCms.createProfile("sRGB")
                    lab_p = ImageCms.createProfile("LAB")
                    rgb_to_lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
                    lab_img = ImageCms.applyTransform(rgb_img, rgb_to_lab)
                    l_chan, a_chan, b_chan = lab_img.split()

                    if exists(join(l, f"{filename}.{img_format}")) is False:
                        #io.imsave(join(lab, f"{filename}.{img_format}"), lab_img)
                        #lab_img.save(join(lab, f"{filename}.png"))
                        l_chan.save(join(l, f"{filename}.{img_format}"))
                    if exists(join(a, f"{filename}.{img_format}")) is False:
                        a_chan.save(join(a, f"{filename}.{img_format}"))
                    if exists(join(b, f"{filename}.{img_format}")) is False:
                        b_chan.save(join(b, f"{filename}.{img_format}"))
                    if exists(join(ds_path, f)) is False:
                        k += 1
                        shutil.move(file, ds_path)

        print(f"{k} images found")
        print("Conversion to LAB done")

    assert test_rate < 1, "test_rate must be less than 1"

    val_rate = 0.1

    splitfolders.ratio(full_ds, output=dest,
                       seed=1337, ratio=(1 - (test_rate + val_rate), val_rate, test_rate), group_prefix=None,
                       move=False)

    print("Splitting into train/val/test...")

    for root, directories, files in os.walk(full_ds):
        for d in directories:
            if d == "train" or d == "val" or d == "test":
                splitted = join(src, "splitted/")
                if exists(splitted) is False:
                    os.makedirs(splitted)
                shutil.move(join(full_ds, d), splitted)

    print("Splitting done")


def npz_images_converter(src, test_rate):
    """Splits generated images (slices) into train/val/test and stores them in .npz file.

            Parameters
            ----------
            src : str
                Path to the slices directory.

            dest : str
                Path to the destination directory.

            test_rate : float
                Test images rate.
            """

    ds_path = join(src, "img_dataset/")

    if exists(ds_path) is False:
        os.makedirs(ds_path)

    file_count = len([name for name in os.listdir(src) if name.endswith(".jpg")])
    print("Converting RGB images to LAB...")

    inputs = []
    targets = []

    for root, _, files in os.walk(src):
        for f in tqdm(files, total=file_count):
            if not f.endswith(".DS_Store"):
                file = join(root, f)
                rgb_img = io.imread(file)
                lab_img = rgb2lab(rgb_img)

                input = lab_img[:,:,0]
                target = lab_img[:,:,1:]
                inputs.append(input)
                targets.append(target)

                if exists(join(ds_path, f)) is False:
                    shutil.move(file, ds_path)

    assert len(inputs) == len(targets), "inputs and targets must be of the same number"

    val_rate = 0.1

    print("Splitting into train/val/test...")

    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=test_rate, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_rate, random_state=42)

    print("Saving in .npz file...")

    filename = join(src, "lab_dataset.npz")
    np.savez(file=filename,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        X_test=X_test, y_test=y_test)

    print("Done !")


def main():
    """This is the main function"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="path to the dataset")
    parser.add_argument("--output", "-o", help="path to the output")
    parser.add_argument("--testrate", "-t", help="part of test set", default=0.2)
    parser.add_argument("--raw", help="splits raw images directly on disk", default=False, action="store_true")
    parser.add_argument("--create_lab", "-c", help="create lab images for train", default=False, action="store_true")

    args = parser.parse_args()

    datapath = args.datapath
    output = args.output
    test_rate = args.testrate
    raw = args.raw
    create_lab = args.create_lab

    if raw:
        raw_splitter(datapath, output, test_rate, create_lab)
    else:
        npz_images_converter(datapath, test_rate)



if __name__ == "__main__":
    main()