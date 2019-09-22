import tensorflow as tf
from satellite_enhancer.model.layer.encoder_block import EncoderBlock


class Discriminator(tf.keras.Model):

    def __init__(self, name='discriminator'):
        super(Discriminator, self).__init__(name)

        self.num_filters = 64

        # Gaussian init with mean 0 and std 0.02 is good for GANS
        self.conv_1 = tf.keras.layers.Conv2D(self.num_filters, kernel_size=3, strides=1, padding='same', name='conv_1',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.lrelu_1 = tf.keras.layers.LeakyReLU(alpha=0.2, name='lrelu_1')

        self.encoder_1 = EncoderBlock(num_filters=self.num_filters, strides=2, name='encoder_1')
        self.encoder_2 = EncoderBlock(num_filters=self.num_filters * 2, strides=1, name='encoder_2')
        self.encoder_3 = EncoderBlock(num_filters=self.num_filters * 2, strides=2, name='encoder_3')
        self.encoder_4 = EncoderBlock(num_filters=self.num_filters * 4, strides=1, name='encoder_4')
        self.encoder_5 = EncoderBlock(num_filters=self.num_filters * 4, strides=2, name='encoder_5')
        self.encoder_6 = EncoderBlock(num_filters=self.num_filters * 8, strides=1, name='encoder_6')
        self.encoder_7 = EncoderBlock(num_filters=self.num_filters * 8, strides=2, name='encoder_7')

        self.flatten = tf.keras.layers.Flatten(name='flatten_1')

        self.dense_1 = tf.keras.layers.Dense(self.num_filters * 16, name='dense_1')
        self.lrelu_2 = tf.keras.layers.LeakyReLU(alpha=0.2, name='lrelu_2')

        self.dropout = tf.keras.layers.Dropout(0.5)

        self.dense_2 = tf.keras.layers.Dense(1, activation='sigmoid', name='dense_2')

        # loss
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def call(self, inputs, training=True):

        x = self.conv_1(inputs)

        x = self.lrelu_1(x)

        x = self.encoder_1(x, training=training)

        x = self.encoder_2(x, training=training)
        x = self.encoder_3(x, training=training)

        x = self.encoder_4(x, training=training)
        x = self.encoder_5(x, training=training)

        x = self.encoder_6(x, training=training)
        x = self.encoder_7(x, training=training)

        x = self.flatten(x)

        x = self.dense_1(x)
        x = self.lrelu_2(x)

        #x = self.dropout(x)

        x = self.dense_2(x)

        return x

    def loss(self, hr_out, sr_out):
        """
        Same
        :param hr_out:
        :param sr_out:
        :return:
        """

        # EPS = 1e-12
        # hr_loss = tf.reduce_mean(tf.math.log(hr_out + EPS))
        # sr_loss = tf.reduce_mean(tf.math.log((1 - sr_out) + EPS))

        hr_loss = self.binary_crossentropy(tf.ones_like(hr_out), hr_out)  # log(y_real)
        sr_loss = self.binary_crossentropy(tf.zeros_like(sr_out), sr_out)  # log(1-y_fake)

        print(f'DiscHR: {hr_loss}')
        print(f'DiscSR: {sr_loss}')

        return (sr_loss + hr_loss) #* 0.5



