import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.losses import MeanSquaredError
from satellite_enhancer.model.layer.residual_block import ResidualBlock
from satellite_enhancer.model.layer.upsample_block import UpsampleBlock
from satellite_enhancer.model.vgg_feature import VGGFeature
import matplotlib.pyplot as plt


class Generator(tf.keras.Model):

    def __init__(self, name='generator'):
        super(Generator, self).__init__(name)

        # TODO: understand better why to use VGG convolution feature?
        self.vgg_feature = VGGFeature().output_layer()

        # generator structure
        self.conv_1 = tf.keras.layers.Conv2D(64, kernel_size=9, strides=1, padding='same', name='conv_1',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.prelu_1 = tf.keras.layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2], name='prelu_1')

        self.res_blocks = list(map(lambda x: ResidualBlock(name=f'residual_{x}'), range(16)))

        self.conv_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', name='conv_2')
        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=0.8, name='bn_1',)

        self.upsample_blocks = list(map(lambda x: UpsampleBlock(num_filters=256, name=f'upsample_{x}'), range(2)))

        self.conv_3 = tf.keras.layers.Conv2D(filters=3, kernel_size=9, strides=1, padding="same", activation='tanh',
                                             name='conv_3')

        # loss
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.mse = MeanSquaredError()

    def call(self, inputs, training=True):
        """
        Build the Generator model
        :param tf.Dataset inputs: dataset input
        :param Boolean training: flag if training
        :return: x
        """

        print(f'Input: {inputs.shape}')

        x = self.conv_1(inputs)
        x = self.prelu_1(x)
        print(x.shape)

        skip_res_x = x

        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x, training)
            print(f'Res_{i}: {x.shape}')

        x = self.conv_2(x)
        x = self.bn_1(x)

        x = tf.keras.layers.add([x, skip_res_x])
        print(x.shape)

        # Using 2 UpSampling Blocks
        for upsample_block in self.upsample_blocks:
            x = upsample_block(x)
            print(f'Upscale_{i}: {x.shape}')

        x = self.conv_3(x)

        print(f'Finale: {x.shape}')

        return x

    def loss(self, hr, sr, sr_output):
        """
        Perceptual loss
        :param hr:
        :param sr:
        :param sr_output:
        :return:
        """
        con_loss = self.content_loss(hr, sr)
        gen_loss = self.adversarial_loss(sr_output)

        print(f'ContentLoss: {con_loss}')
        print(f'GenLoss: {con_loss}')

        # perceptual loss
        perc_loss = con_loss + 0.001 * gen_loss

        return perc_loss

    def adversarial_loss(self, disc_sr):
        EPS = 1e-12
        # numerical instability
        advr_loss = tf.reduce_sum(-tf.math.log(disc_sr + EPS))

        test = self.binary_crossentropy(tf.ones_like(disc_sr), disc_sr)

        print(f'AdvrSR: {advr_loss} VS {test}')

        return advr_loss

    def denormalize(self, img):
        #return tf.divide(tf.add(img, 1), 2.)
        return tf.cast(tf.cast(tf.multiply(tf.add(img, 1), 127.5), dtype=tf.uint32), dtype=tf.float32)

    def content_loss(self, hr, sr):

        import numpy as np

        hr = self.denormalize(hr)
        sr = self.denormalize(sr)

        # print(np.max(hr[0].numpy()))
        # print(np.max(sr[0].numpy()))
        # print(np.min(hr[0].numpy()))
        # print(np.min(sr[0].numpy()))
        #
        # plt.imshow(hr[0])
        # plt.show()
        # plt.imshow(sr[0])
        # plt.show()

        sr = preprocess_input(sr)
        hr = preprocess_input(hr)

        # plt.imshow(hr[0])
        # plt.show()
        # plt.imshow(sr[0])
        # plt.show()
        #
        # print(np.max(hr[0].numpy()))
        # print(np.max(sr[0].numpy()))
        # print(np.min(hr[0].numpy()))
        # print(np.min(sr[0].numpy()))

        sr_features = self.vgg_feature(sr)
        hr_features = self.vgg_feature(hr)

        return self.mse(hr_features, sr_features) / 12.75
