import tensorflow as tf

LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)  # Having the discriminator predict ones on generated image would be ideal for generator

    l1_loss = tf.reduce_mean(tf.abs(target-gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)  # Taking how realistic the image is into account, however mainly prioritizing how close it is to the target --> Conditional Generative Adversial Network

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)   # Predicts 1 for real, 0 for fake

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)   # Should ideally (for the discriminator) predict only 0:s on generated images

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss