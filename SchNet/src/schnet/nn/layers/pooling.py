import tensorflow as tf


class PoolSegments(tf.keras.layers.Layer):
    def __init__(self, mode='sum'):
        super().__init__()
        if mode == 'sum':
            self._reduce = tf.math.segment_sum
        elif mode == 'mean':
            self._reduce = tf.math.segment_mean

    def call(self, x, segs):
        x = Squeeze()(x)
        segs = Squeeze()(segs)
        y = self._reduce(x, segs)
        y = Expand()(y)
        return y

class Squeeze(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        y = tf.keras.backend.squeeze(x, 0)
        return y


class Expand(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        y = tf.keras.backend.expand_dims(x, 0)
        return y