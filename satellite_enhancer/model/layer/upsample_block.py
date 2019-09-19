import tensorflow as tf


class UpsampleBlock(tf.keras.Model):

    def __init__(self):
        super(UpsampleBlock, self).__init__()

    def call(self, inputs, num_filters=64):

        x = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, strides=1, padding="same")(inputs)
        x = tf.keras.layers.UpSampling2D(size=2)(x)
        x = tf.keras.activations.LeakyReLU(alpha=0.2)(x)

        return x
