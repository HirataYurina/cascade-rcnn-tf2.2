# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:fpn.py
# software: PyCharm

import tensorflow as tf
import tensorflow.keras.layers as layers


class FPN(tf.keras.Model):

    def __init__(self, out_channel=256):
        """
        特征金字塔
        ------------------
        :param out_channel: 输出通道数量
        """
        super(FPN, self).__init__()

        self.out_channel = out_channel

        # 卷积层
        self.fpn_c2p2 = layers.Conv2D(out_channel, 1, kernel_initializer='he_normal', name='fpn_c2p2')
        self.fpn_c3p3 = layers.Conv2D(out_channel, 1, kernel_initializer='he_normal', name='fpn_c3p3')
        self.fpn_c4p4 = layers.Conv2D(out_channel, 1, kernel_initializer='he_normal', name='fpn_c4p4')
        self.fpn_c5p5 = layers.Conv2D(out_channel, 1, kernel_initializer='he_normal', name='fpn_c5p5')

        # 上采样层
        self.upsampled_p5 = layers.UpSampling2D((2, 2), name='upsampled_p5')
        self.upsampled_p4 = layers.UpSampling2D((2, 2), name='upsampled_p4')
        self.upsampled_p3 = layers.UpSampling2D((2, 2), name='upsampled_p3')

        # feature输出层
        self.p5 = layers.Conv2D(out_channel, 3, 1, padding='same', kernel_initializer='he_normal', name='p5')
        self.p4 = layers.Conv2D(out_channel, 3, 1, padding='same', kernel_initializer='he_normal', name='p4')
        self.p3 = layers.Conv2D(out_channel, 3, 1, padding='same', kernel_initializer='he_normal', name='p3')
        self.p2 = layers.Conv2D(out_channel, 3, 1, padding='same', kernel_initializer='he_normal', name='p2')

        # pooling层
        self.p6 = layers.MaxPool2D(2, name='p6')

    def call(self, inputs, training=None, mask=None):
        # resnet的多尺寸输出
        c2, c3, c4, c5 = inputs

        p5 = self.fpn_c5p5(c5)
        p4 = self.fpn_c4p4(c4) + self.upsampled_p5(p5)
        p3 = self.fpn_c3p3(c3) + self.upsampled_p4(p4)
        p2 = self.fpn_c2p2(c2) + self.upsampled_p3(p3)

        # 提取特征
        p5 = self.p5(p5)
        p4 = self.p4(p4)
        p3 = self.p3(p3)
        p2 = self.p2(p2)

        # pooling
        p6 = self.p6(p5)

        return [p2, p3, p4, p5, p6]

    def compute_output_shape(self, input_shape):
        # input_shape是TensorShape
        c2_shape, c3_shape, c4_shape, c5_shape = input_shape

        c2_shape = c2_shape.as_list()
        c3_shape = c3_shape.as_list()
        c4_shape = c4_shape.as_list()
        c5_shape = c5_shape.as_list()

        c2_shape[-1] = self.out_channel
        c3_shape[-1] = self.out_channel
        c4_shape[-1] = self.out_channel
        c5_shape[-1] = self.out_channel

        c6_shape = [c5_shape[0], c5_shape[1] // 2, c5_shape[2] // 2, c5_shape[3]]

        return [tf.TensorShape(c2_shape), tf.TensorShape(c3_shape), tf.TensorShape(c4_shape),
                tf.TensorShape(c5_shape), tf.TensorShape(c6_shape)]


if __name__ == '__main__':

    inp_c2 = tf.keras.Input(shape=(256, 256, 256))
    inp_c3 = tf.keras.Input(shape=(128, 128, 512))
    inp_c4 = tf.keras.Input(shape=(64, 64, 1024))
    inp_c5 = tf.keras.Input(shape=(32, 32, 2048))

    fpn = FPN(out_channel=256)

    out_p2, out_p3, out_p4, out_p5, out_p6 = fpn([inp_c2, inp_c3, inp_c4, inp_c5])

    fpn.summary()
