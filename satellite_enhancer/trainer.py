import tensorflow as tf
import numpy as np
import os
from satellite_enhancer.model.generator import Generator
from satellite_enhancer.model.discriminator import Discriminator
import matplotlib.pyplot as plt


class Trainer(object):

    def __init__(self, generator: Generator, discriminator: Discriminator):

        self.generator = generator
        self.generator.load_weights('../trained_models/generator/gen_1')

        self.discriminator = discriminator
        self.discriminator.load_weights('../trained_models/discriminator/disc_1')

        # discriminator should learn then generator
        self.generator_optimizer = tf.keras.optimizers.Adam(lr=1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(lr=1e-6)

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

    def train(self, train_dataset, num_epochs=100):

        with self.summary_writer.as_default():

            for epoch in range(num_epochs):
                print(f'EPOCHE: {epoch}')

                pl, dl = 0, 0
                num_setps = 0
                for lr, hr in train_dataset.take(50):
                    print(f'{num_setps}/{epoch}')
                    num_setps += 1

                    p_l, d_l = self.train_step(lr, hr)
                    pl += p_l
                    dl += d_l

                # tensorboard
                tf.summary.scalar('perceptual_loss', pl/num_setps, step=self.generator_optimizer.iterations)
                tf.summary.scalar('discriminator_loss', dl/num_setps, step=self.discriminator_optimizer.iterations)

                print(f'{epoch}/{num_epochs}, perceptual loss = {pl:.3f}, discriminator loss = {dl:.3f}')

                self.generator.save_weights('../trained_models/generator/gen_1', save_format='tf')
                self.discriminator.save_weights('../trained_models/discriminator/disc_1', save_format='tf')

    #@tf.function
    def train_step(self, lr, hr):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            lr = self.normalize(lr)
            hr = self.normalize(hr)

            sr = self.generator(lr, training=True)
            #self.generator.summary()

            # plt.imshow(self.denormalize(hr[0]))
            # plt.show()
            # # plt.imshow(self.denormalize(lr[0]))
            # # plt.show()
            # plt.imshow(self.denormalize(sr[0]))
            # plt.show()

            hr_output = self.discriminator(hr, training=True)
            sr_output = self.discriminator(sr, training=True)
            #self.discriminator.summary()

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

    def flip_labels(self, label, prob=0.05):
        pass

    def normalize(sel, img):
        """Normalize image to [-1,1]"""

        img = tf.cast(img, tf.float32)/127.5 - tf.ones_like(img, dtype=np.float32)
        #n_img = np.divide(img.astype(np.float32), 127.5) - np.ones_like(img, dtype=np.float32)

        return img

    def denormalize(self, img):
        # return tf.divide(tf.add(img, 1), 2.)
        return tf.cast(tf.multiply(tf.add(img, 1), 127.5), dtype=tf.uint32)

if __name__ == '__main__':

    from satellite_enhancer.dataset.divk2_dataset import DIV2K
    from satellite_enhancer.dataset.satellite_dataset import SatelliteDataset

    #div2k_train = DIV2K(scale=4, subset='train', downgrade='bicubic')
    satelitte_train = SatelliteDataset(scale=4, images_dir='.satellite/images')

    train_ds = satelitte_train.dataset(batch_size=16, random_transform=True)

    trainer = Trainer(Generator(), Discriminator())

    # for x in train_ds:
    #     for image in zip(x[0], x[1]):
    #         print(image[0].shape)
    #         print(image[1].shape)

    trainer.train(train_ds, num_epochs=500)
