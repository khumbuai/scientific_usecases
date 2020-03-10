import numpy as np
import tensorflow as tf
from schnet.nn.layers.pooling import Squeeze,Expand



class RBFExpansion(tf.keras.layers.Layer):
    def __init__(self, low, high, gap, dim=1, name=None):
        super().__init__()
        self.low = low
        self.high = high
        self.gap = gap
        self.dim = dim

        xrange = high - low
        self.centers = np.linspace(low, high, int(np.ceil(xrange / gap)))
        self.centers = self.centers[:, np.newaxis]
        self.n_centers = len(self.centers)
        self.fan_out = self.dim * self.n_centers

    def call(self, d):
        #cshape = tf.shape(d)
        d = Squeeze()(d)
        CS = d.get_shape()
        centers = self.centers.reshape((1, -1)).astype(np.float32)
        d -= tf.constant(centers)
        rbf = tf.exp(-(d**2) / self.gap)
        # rbf = tf.reshape(rbf, (
        #     cshape[0], cshape[1], cshape[2],
        #     self.dim * centers.shape[-1]))
        rbf.set_shape([CS[0], self.fan_out])
        rbf = Expand()(rbf)
        return rbf
