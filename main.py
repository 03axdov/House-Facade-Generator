import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_processing import load_image_train, load_image_test, PATH
from models import downsample, upsample, Generator, Discriminator
from losses import generator_loss, discriminator_loss


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
    # tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64, to_file="generator.png")

    # plt.imshow(input)
    # plt.show()

    # gen_output = generator(input[tf.newaxis, ...], training=False)
    # plt.imshow(gen_output[0, ...])
    # plt.show()

    discriminator = Discriminator()
    # tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64, to_file="discriminator.png")

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_dir = "./training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator
    )

    def generate_images(model, test_input, target):
        if len(test_input.shape) != 4:
            test_input = test_input[tf.newaxis, ...]
        if len(target.shape) != 4:
            target = target[tf.newaxis, ...]
        prediction = model(test_input, training=True)
        plt.figure(figsize=(11,11))

        display_list = [test_input[0], target[0], prediction[0]]
        title = ["Input image", "Ground Truth", "Predicted Image"]

        for i in range(3):
            plt.subplot(1,3,i+1)
            plt.title(title[i])
            plt.imshow(display_list[i]*0.5+0.5)
            plt.axis("off")
        plt.show()

    # for example_input, example_target in test_dataset.take(1):
    #     generate_images(generator, example_input[tf.newaxis, ...], example_target[tf.newaxis, ...])

    log_dir = "logs/"

    import datetime
    summary_writer = tf.summary.create_file_writer(
        log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    @tf.function
    def train_step(input_image, target, step):
        if len(input_image) != 4:
            input_image = input_image[tf.newaxis, ...]
        if len(target) != 4:
            target = target[tf.newaxis, ...]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)

            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(
            gen_total_loss, generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )

        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step//1000)


    import time
    def fit(train_ds, test_ds, steps):
        example_input, example_target = next(iter(test_ds.take(1)))
        start  = time.time()

        for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
            if step % 1000 == 0:
                if step != 0:
                    print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

                start = time.time()

                generate_images(generator, example_input, target)
                print(f"Step: {step//1000}k")

            train_step(input_image, target, step)

            if (step+1) % 10 == 0:
                print('.', end='', flush=True)

            if (step+1) % 5000 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)


    fit(train_dataset, test_dataset, steps=40000)


if __name__ == "__main__":
    main()