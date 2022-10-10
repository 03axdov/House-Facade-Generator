import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_processing import load_image_train, load_image_test, PATH
from models import downsample, upsample, Generator


def main():

    BUFFER_SIZE = 400
    BATCH_SIZE = 1  # Seemingly produces ideal results in the original paper

    train_dataset = tf.data.Dataset.list_files(str(PATH / "train/*.jpg"))
    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset.shuffle(BUFFER_SIZE)
    train_dataset.batch(BATCH_SIZE)

    try:
        test_dataset = tf.data.Dataset.list_files(str(PATH / "test/*.jpg"))
    except tf.errors.InvalidArgumentError:
        test_dataset = tf.data.Dataset.list_files(str(PATH / "val/*.jpg"))

    test_dataset = test_dataset.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset.batch(BATCH_SIZE)

    down_model = downsample(3, 4)

    for input, target in train_dataset.take(1):
        break

    down_result = down_model(tf.expand_dims(input, 0))
    print(f"down_result.shape: {down_result.shape}")    # (1, 128, 128, 3)

    up_model = upsample(3,4)
    up_result = up_model(down_result)
    print(f"up_result.shape: {up_result.shape}")    # (1, 256, 256, 3)

    generator = Generator()
    # tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

    plt.imshow(input)
    plt.show()

    gen_output = generator(input[tf.newaxis, ...], training=False)
    plt.imshow(gen_output[0, ...])
    plt.show()


if __name__ == "__main__":
    main()