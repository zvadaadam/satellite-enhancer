import tensorflow as tf
from tensorflow.python.keras.applications.vgg19 import VGG19


class VGGFeature:

    def __init__(self):
        # TODO: need out own trained VGG for satelliate imagery
        self.vgg = VGG19(input_shape=(None, None, 3), weights='imagenet', include_top=False)
        self.vgg.trainable = False

        for layer in self.vgg.layers:
            layer.trainable = False

    def output_layer(self, output_layer):

        loss_model = tf.keras.Model(inputs=self.vgg.input, outputs=self.vgg.layers[output_layer].output)
        loss_model.trainable = False

        return loss_model


