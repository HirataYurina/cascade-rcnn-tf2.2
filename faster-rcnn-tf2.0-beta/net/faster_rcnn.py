# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:faster_rcnn.py
# software: PyCharm

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

# loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.NONE)
#
# y_pred = tf.random.normal((10,), dtype=tf.float32)
# print(y_pred * tf.constant(10, dtype=tf.int32))

a = tf.constant([1, 2, 4])
a = tf.expand_dims(a, axis=0)
# print(a[0:11])

b = tf.constant([[1, 2, 3, 4], [4, 3, 2, 1]])
# b = tf.expand_dims(b, axis=0)

# top_k = tf.nn.top_k(b, 3)
# print(top_k)

# sort = tf.argmax(b, axis=1)
# print(sort)

print(tf.split(b, 4, axis=1))

def test():
    return [1, 2, 3]

print(test())