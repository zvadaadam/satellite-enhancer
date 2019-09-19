import tensorflow as tf


class EncoderBlock(tf.keras.Model):

    def __init__(self):
        super(EncoderBlock, self).__init__()

    def call(self, inputs, num_filters=64, strides=1, batchnorm=True, training=True):

        x = tf.keras.layers.Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(inputs)

        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x, training)

        return tf.keras.layers.LeakyReLU(alpha=0.2)(x)
