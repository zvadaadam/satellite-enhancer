import tensorflow as tf
from satellite_enhancer.model.layer.encoder_block import EncoderBlock


class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__(self)

        self.num_filters = 64

        self.conv_1 = tf.keras.layers.Conv2D(self.num_filters, kernel_size=3, strides=1, padding='same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.lrelu_1 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.encoder_1 = EncoderBlock()
        self.encoder_2 = EncoderBlock()
        self.encoder_3 = EncoderBlock()
        self.encoder_4 = EncoderBlock()
        self.encoder_5 = EncoderBlock()
        self.encoder_6 = EncoderBlock()
        self.encoder_7 = EncoderBlock()

        self.flatten = tf.keras.layers.Flatten()

        self.dense_1 = tf.keras.layers.Dense(1024)
        self.lrelu_2 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.dense_2 = tf.keras.layers.Dense(1, activation='sigmoid')

        # loss
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def call(self, inputs, training=True):

        x = self.conv_1(inputs)

        x = self.lrelu_1(x)

        x = self.encoder_1(x, num_filters=self.num_filters, strides=2, training=training)

        x = self.encoder_2(x, num_filters=self.num_filters*2, training=training)
        x = self.encoder_3(x, num_filters=self.num_filters*2, strides=2, training=training)

        x = self.encoder_4(x, num_filters=self.num_filters * 4, training=training)
        x = self.encoder_5(x, num_filters=self.num_filters * 4, strides=2, training=training)

        x = self.encoder_6(x, num_filters=self.num_filters * 8, training=training)
        x = self.encoder_7(x, num_filters=self.num_filters * 8, strides=2, training=training)

        x = self.flatten(x)

        x = self.dense_1(x)
        x = self.lrelu_2(x)

        x = self.dense_2(x)

        return x

    def loss(self, hr_out, sr_out):

        # discrim_fake_loss = tf.log(1 - discrim_fake_output + FLAGS.EPS)
        # discrim_real_loss = tf.log(discrim_real_output + FLAGS.EPS)
        #
        # discrim_loss = tf.reduce_mean(-(discrim_fake_loss + discrim_real_loss))

        hr_loss = self.binary_crossentropy(tf.ones_like(hr_out), hr_out)  # log(y_real)
        sr_loss = self.binary_crossentropy(tf.zeros_like(sr_out), sr_out)  # log(1-y_fake)

        return hr_loss + sr_loss



