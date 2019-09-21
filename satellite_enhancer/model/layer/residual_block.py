import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, name='residual'):
        super(ResidualBlock, self).__init__(name)

        self.conv_1 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=0.8)

        self.prelu_1 = tf.keras.layers.PReLU(shared_axes=[1, 2])

        self.conv_2 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        self.bn_2 = tf.keras.layers.BatchNormalization(momentum=0.8)

    def call(self, inputs, training=True):
        #x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")

        x = self.conv_1(inputs, training=training)

        #x = self.bn_1(x, training=training)
        x = self.bn_1(x)

        x = self.prelu_1(x)

        #x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")

        x = self.conv_2(x, training=training)

        #x = self.bn_2(x, training=training)
        x = self.bn_2(x)

        x = tf.add(x, inputs)

        return x
