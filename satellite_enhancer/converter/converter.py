from satellite_enhancer.model.generator import Generator
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class Converter:
    def __init__(self):
        self.generator = Generator()
        self.generator.load_weights('../../trained_models/generator/gen_1')

    def convert(self, imagePath, outputName):
        img_raw = tf.io.read_file(imagePath)
        # cropWindow = tf.Variable([0,0,32,32])
        # image = tf.expand_dims(tf.image.decode_and_crop_jpeg(img_raw, crop_window=cropWindow), 0)
        image = tf.expand_dims(tf.image.decode_jpeg(img_raw), 0)
        # plt.imshow(image.numpy()[0])
        # plt.show()
        image = self.normalize(image)
        superResImg = self.generator.call(image, False)
        superResImg = self.denormalize(superResImg)
        # plt.imshow(superResImg.numpy()[0])
        # plt.show()

        enc = tf.image.encode_jpeg(superResImg.numpy()[0])
        fname = tf.constant(outputName)
        tf.io.write_file(fname, enc)


    def denormalize(self, img):
        return tf.cast(tf.multiply(tf.add(img, 1), 127.5), dtype=tf.uint8)

    def normalize(self, img):
        img = tf.cast(img, tf.float32)/127.5 - tf.ones_like(img, dtype=np.float32)
        return img

converter = Converter()
for i in range(0, 12):
    converter.convert(f"../.satellite/images/LR/tile1-{i}.jpg", f"./tile1-{i}-superres.jpg")
