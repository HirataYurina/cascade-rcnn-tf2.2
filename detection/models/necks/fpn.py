# -*- coding:utf-8 -*-
# author:栗山未来ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:aaa.py
# software: PyCharm


import tensorflow as tf

layers = tf.keras.layers


class FPN(tf.keras.Model):
    def __init__(self, out_channels=256, **kwargs):
        '''Feature Pyramid Networks
        
        Attributes
        ---
            out_channels: int. the channels of pyramid feature maps.
        '''
        super(FPN, self).__init__(**kwargs)

        self.out_channels = out_channels

        self.fpn_c2p2 = layers.Conv2D(out_channels, (1, 1), name='fpn_c2p2')
        self.fpn_c3p3 = layers.Conv2D(out_channels, (1, 1), name='fpn_c3p3')
        self.fpn_c4p4 = layers.Conv2D(out_channels, (1, 1), name='fpn_c4p4')
        self.fpn_c5p5 = layers.Conv2D(out_channels, (1, 1), name='fpn_c5p5')

        self.fpn_p3upsampled = layers.UpSampling2D(size=(2, 2), name='fpn_p3upsampled')
        self.fpn_p4upsampled = layers.UpSampling2D(size=(2, 2), name='fpn_p4upsampled')
        self.fpn_p5upsampled = layers.UpSampling2D(size=(2, 2), name='fpn_p5upsampled')

        self.fpn_p2 = layers.Conv2D(out_channels, (3, 3), padding='SAME', name='fpn_p2')
        self.fpn_p3 = layers.Conv2D(out_channels, (3, 3), padding='SAME', name='fpn_p3')
        self.fpn_p4 = layers.Conv2D(out_channels, (3, 3), padding='SAME', name='fpn_p4')
        self.fpn_p5 = layers.Conv2D(out_channels, (3, 3), padding='SAME', name='fpn_p5')

        self.fpn_p6 = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='fpn_p6')

    def __call__(self, inputs, training=True):
        C2, C3, C4, C5 = inputs

        P5 = self.fpn_c5p5(C5)
        P4 = self.fpn_c4p4(C4) + self.fpn_p5upsampled(P5)
        P3 = self.fpn_c3p3(C3) + self.fpn_p4upsampled(P4)
        P2 = self.fpn_c2p2(C2) + self.fpn_p3upsampled(P3)

        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = self.fpn_p2(P2)
        P3 = self.fpn_p3(P3)
        P4 = self.fpn_p4(P4)
        P5 = self.fpn_p5(P5)

        # subsampling from P5 with stride of 2.
        P6 = self.fpn_p6(P5)

        return [P2, P3, P4, P5, P6]

    def compute_output_shape(self, input_shape):
        C2_shape, C3_shape, C4_shape, C5_shape = input_shape

        C2_shape, C3_shape, C4_shape, C5_shape = \
            C2_shape.as_list(), C3_shape.as_list(), C4_shape.as_list(), C5_shape.as_list()

        C6_shape = [C5_shape[0], C5_shape[1] // 2, C5_shape[2] // 2, self.out_channels]

        C2_shape[-1] = self.out_channels
        C3_shape[-1] = self.out_channels
        C4_shape[-1] = self.out_channels
        C5_shape[-1] = self.out_channels

        return [tf.TensorShape(C2_shape),
                tf.TensorShape(C3_shape),
                tf.TensorShape(C4_shape),
                tf.TensorShape(C5_shape),
                tf.TensorShape(C6_shape)]


if __name__ == '__main__':
    C2 = tf.random.normal((2, 200, 200, 256))
    C3 = tf.random.normal((2, 100, 100, 512))
    C4 = tf.random.normal((2, 50, 50, 1024))
    C5 = tf.random.normal((2, 25, 25, 2048))

    fpn = FPN()

    P2, P3, P4, P5, P6 = fpn([C2, C3, C4, C5])

    print('P2 shape:', P2.shape.as_list())
    print('P3 shape:', P3.shape.as_list())
    print('P4 shape:', P4.shape.as_list())
    print('P5 shape:', P5.shape.as_list())
    print('P6 shape:', P6.shape.as_list())
