import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19


class VGGFeature:

    def __init__(self):
        # TODO: need out own trained VGG for satelliate imagery
        self.vgg = VGG19(input_shape=(96, 96, 3), weights='imagenet', include_top=False)
        self.vgg.trainable = False

        for layer in self.vgg.layers:
            layer.trainable = False

    def output_layer(self, output_layer='block5_conv4'):

        loss_model = tf.keras.Model(inputs=self.vgg.input,
                                    outputs=self.vgg.get_layer(output_layer).output)
        loss_model.trainable = False

        return loss_model


