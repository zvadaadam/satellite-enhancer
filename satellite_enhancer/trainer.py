import tensorflow as tf
import numpy as np
import os
from satellite_enhancer.model.generator import Generator
from satellite_enhancer.model.discriminator import Discriminator
import matplotlib.pyplot as plt


class Trainer(object):

    def __init__(self, generator: Generator, discriminator: Discriminator):

        self.generator = generator
        self.discriminator = discriminator

        self.generator_optimizer = tf.keras.optimizers.Adam(0.001)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.001)

        self.summary_writer = tf.summary.create_file_writer(os.path.join('summaries', 'train'))

        # self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
        #                                       psnr=tf.Variable(-1.0),
        #                                       optimizer=Adam(learning_rate),
        #                                       model=model)
        # self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
        #                                                      directory=checkpoint_dir,
        #                                                      max_to_keep=3)

        #self.restore()

    def high_level_train(self):
        pass

    def train(self, train_dataset, num_steps=2000):

        with self.summary_writer.as_default():
            step = 0
            for lr, hr in train_dataset.take(num_steps):
                step += 1

                pl, dl = self.train_step(lr, hr)

                # tensorboard
                tf.summary.scalar('perceptual_loss', pl, step=self.generator_optimizer.iterations)
                tf.summary.scalar('discriminator_loss', dl, step=self.discriminator_optimizer.iterations)

                print(f'{step}/{num_steps}, perceptual loss = {pl:.3f}, discriminator loss = {dl:.3f}')

    #@tf.function
    def train_step(self, lr, hr):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            lr = self.normalize(lr)
            hr = self.normalize(hr)

            sr = self.generator(lr, training=True)

            # plt.imshow(hr[0])
            # plt.show()
            # plt.imshow(lr[0])
            # plt.show()
            # plt.imshow(sr[0])
            # plt.show()

            hr_output = self.discriminator(hr, training=True)
            sr_output = self.discriminator(sr, training=True)

            print(f'HR_disc: {hr_output.numpy()}')
            print(f'SR_disc: {sr_output.numpy()}')

            perc_loss = self.generator.loss(hr, sr, sr_output)

            disc_loss = self.discriminator.loss(hr_output, sr_output)

        gradients_of_generator = gen_tape.gradient(perc_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )

        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )

        return perc_loss, disc_loss

    def normalize(sel, img):
        """Normalize image to [-1,1]"""

        img = tf.cast(img, tf.float32)/127.5 - tf.ones_like(img, dtype=np.float32)
        #n_img = np.divide(img.astype(np.float32), 127.5) - np.ones_like(img, dtype=np.float32)

        return img

    def denormalize(self, img):
        return ((img + 1) * 127.5).astype(np.uint8)


if __name__ == '__main__':

    from satellite_enhancer.dataset.divk2_dataset import DIV2K
    div2k_train = DIV2K(scale=4, subset='train', downgrade='bicubic')

    train_ds = div2k_train.dataset(batch_size=16, random_transform=True)

    trainer = Trainer(Generator(), Discriminator())

    # for x in train_ds:
    #     for image in zip(x[0], x[1]):
    #         print(image[0].shape)
    #         print(image[1].shape)

    trainer.train(train_ds, num_steps=10000)
