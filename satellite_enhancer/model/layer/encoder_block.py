import tensorflow as tf


class EncoderBlock(tf.keras.layers.Layer):

    def __init__(self, num_filters, strides, name='encoder'):
        super(EncoderBlock, self).__init__(name)

        self.conv_1 = tf.keras.layers.Conv2D(num_filters, kernel_size=3, strides=strides, padding='same',
                                             name='conv_1')

        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=0.8)

        self.lrelu_1 = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, inputs, bn=True, training=True):

        x = self.conv_1(inputs)

        if bn:
            #x = self.bn_1(x, training)
            x = self.bn_1(x)

        x = self.lrelu_1(x)

        return x
