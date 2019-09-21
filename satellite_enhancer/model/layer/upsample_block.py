import tensorflow as tf


class UpsampleBlock(tf.keras.layers.Layer):

    def __init__(self, num_filters=256, name='upsample'):
        super(UpsampleBlock, self).__init__(name)

        self.conv = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=3, strides=1, padding='same',
                                           name='conv')

        # self.pixel_shuffle = tf.keras.layers.UpSampling2D(size=2)
        self.pixel_shuffle = tf.keras.layers.Lambda(
            lambda x: tf.nn.depth_to_space(x, 2)
        )

        self.prelu = tf.keras.layers.PReLU(shared_axes=[1, 2])

        # TODO: TRY UpSampling2D
        # model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
        # model = LeakyReLU(alpha = 0.25)(model)
        # model = UpSampling2D(size = 3)(model)
        # model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
        # model = LeakyReLU(alpha = 0.3)(model)

    def call(self, inputs):

        x = self.conv(inputs)

        x = self.pixel_shuffle(x)

        x = self.prelu(x)

        return x
