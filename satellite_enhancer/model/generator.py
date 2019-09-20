import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import MeanSquaredError
from satellite_enhancer.model.layer.residual_block import ResidualBlock
from satellite_enhancer.model.layer.upsample_block import UpsampleBlock
from satellite_enhancer.model.vgg_feature import VGGFeature


class Generator(tf.keras.Model):

    def __init__(self, vgg_output_layer=20):
        super(Generator, self).__init__(self)

        # TODO: understand better why to use VGG convolution feature?
        self.vgg_feature = VGGFeature().output_layer(vgg_output_layer)

    def call(self, inputs, training=True):
        """
        Build the Generator model
        :param tf.Dataset inputs: dataset input
        :param Boolean training: flag if training
        :return: x
        """

        print(inputs.shape)
        x = tf.keras.layers.Conv2D(64, kernel_size=9, strides=1, padding='same',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02))(inputs)

        x = tf.keras.layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(x)
        print(x.shape)

        skip_res_x = x

        for i in range(16):
            x = ResidualBlock()(x, training)
            print(f'Res_{i}: {x.shape}')

        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.5)(x)

        x = tf.keras.layers.add([x, skip_res_x])
        print(x.shape)

        # Using 2 UpSampling Blocks
        for i in range(2):
            x = UpsampleBlock()(x, 256)
            print(f'Upscale_{i}: {x.shape}')

        x = tf.keras.layers.Conv2D(filters=3, kernel_size=9, strides=1, padding="same", activation='tanh')(x)

        print(f'Finale: {x.shape}')

        return x

    @tf.function
    def loss(self, hr, sr, sr_output):
        """
        Perceptual loss
        :param hr:
        :param sr:
        :param sr_output:
        :return:
        """
        con_loss = self.content_loss(hr, sr)
        gen_loss = self.generator_loss(sr_output)

        # perceptual loss
        perc_loss = con_loss + 0.001 * gen_loss

        return perc_loss

    @tf.function
    def generator_loss(self, sr):
        return tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(sr), sr)

    @tf.function
    def content_loss(self, hr, sr):

        sr = preprocess_input(sr)
        hr = preprocess_input(hr)

        # TODO: why 12.75?
        sr_features = self.vgg_feature(sr) / 12.75
        hr_features = self.vgg_feature(hr) / 12.75

        return MeanSquaredError()(hr_features, sr_features)






