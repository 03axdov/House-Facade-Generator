import tensorflow as tf

IMG_HEIGHT, IMG_WIDTH = 256, 256


def resize(input_image, target_image, height, width):
    input_image = tf.image.resize(input_image, (height, width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    target_image = tf.image.resize(target_image, (height, width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, target_image


def random_crop(input_image, target_image):
    stacked_image = tf.stack([input_image, target_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )

    return cropped_image[0], cropped_image[1]


def normalize(input_image, target_image):
    input_image = (input_image / 127.5) - 1 # Range (-1, 1)
    target_image = (target_image / 127.5) - 1

    return input_image, target_image


@tf.function
def random_jitter(input_image, target_image):
    input_image, target_image = resize(input_image, target_image, 286, 286)

    input_image, target_image = random_crop(input_image, target_image)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)

    return input_image, target_image