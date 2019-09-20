import tensorflow as tf
from satellite_enhancer.model.generator import Generator
from satellite_enhancer.model.discriminator import Discriminator


class Trainer(object):

    def __init__(self, generator: Generator, discriminator: Discriminator):

        self.generator = generator
        self.discriminator = discriminator

    def train(self, train_dataset, num_steps=2000):

        step = 0
        for lr, hr in train_dataset.take(num_steps):
            step += 1

            pl, dl = self.train_step(lr, hr)

            print(f'{step}/{num_steps}, perceptual loss = {pl:.3f}, discriminator loss = {dl:.3f}')


    @tf.function
    def train_step(self, lr, hr):

        generator_optimizer = tf.keras.optimizers.Adam(0.001)
        discriminator_optimizer = tf.keras.optimizers.Adam(0.001)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.generator(lr, training=True)

            hr_output = self.discriminator(hr, training=True)
            sr_output = self.discriminator(sr, training=True)

            perc_loss = self.generator.loss(hr, sr, sr_output)

            disc_loss = self.discriminator.loss(hr_output, sr_output)

        gradients_of_generator = gen_tape.gradient(perc_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return perc_loss, disc_loss

if __name__ == '__main__':


    import matplotlib.pyplot as plt

    from satellite_enhancer.dataset.divk2_dataset import DIV2K
    div2k_train = DIV2K(scale=2, subset='train', downgrade='bicubic')

    train_ds = div2k_train.dataset(batch_size=16, random_transform=True)

    trainer = Trainer(Generator(), Discriminator())

    # for x in train_ds:
    #     for image in zip(x[0], x[1]):
    #         print(image[0].shape)
    #         print(image[1].shape)

    trainer.train(train_ds, num_steps=100)
