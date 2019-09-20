import tensorflow as tf
from satellite_enhancer.model.layer.encoder_block import EncoderBlock

class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__(self)

    def call(self, inputs, training=True):

        x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.02))(inputs)

        x = tf.keras.layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])(x)

        num_filters = 64

        x = EncoderBlock()(x, num_filters=num_filters, strides=2, training=training)

        x = EncoderBlock()(x, num_filters=num_filters*2, training=training)
        x = EncoderBlock()(x, num_filters=num_filters*2, strides=2, training=training)

        x = EncoderBlock()(x, num_filters=num_filters * 4, training=training)
        x = EncoderBlock()(x, num_filters=num_filters * 4, strides=2, training=training)

        x = EncoderBlock()(x, num_filters=num_filters * 8, training=training)
        x = EncoderBlock()(x, num_filters=num_filters * 8, strides=2, training=training)

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        return x

    @tf.function
    def loss(self, hr_out, sr_out):

        hr_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(hr_out), hr_out)
        sr_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(tf.ones_like(sr_out), sr_out)

        return hr_loss + sr_loss



