import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

from utils import *


def load_input_target_images(image_file):

    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)

    w = image.shape[1]
    w = w // 2
    input_image = image[:, w:, :]
    target_image = image[:, :w, :]

    input_image = tf.cast(input_image, tf.float32)
    target_image = tf.cast(target_image, tf.float32)

    return input_image, target_image


def load_data(debug = False):

    dataset_name = "facades"
    URL = f"http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz"

    path_to_zip = tf.keras.utils.get_file(
        fname=f"{dataset_name}.tar.gz",
        extract=True,
        origin=URL
    )

    path_to_zip = Path(path_to_zip)
    PATH = path_to_zip.parent/dataset_name

    if debug:
        # print(list(PATH.parent.iterdir()))

        sample_image = tf.io.read_file(str(PATH / "train/1.jpg"))
        sample_image = tf.io.decode_jpeg(sample_image)
        print(sample_image.shape)   # (256, 512, 3) --> Consists of two (256,256,3) images

        # plt.figure()
        # plt.imshow(sample_image)
        # plt.show()

        input_image, target_image = load_input_target_images(str(PATH / "train/47.jpg"))

        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(input_image / 255.0)
        plt.title("Input")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(target_image / 255.0)
        plt.title("Target")
        plt.axis("off")

        plt.show()


if __name__ == "__main__":
    load_data(debug=True)