import tensorflow as tf


#from .dense import Dense
from .pooling import PoolSegments

class CFConv(tf.keras.layers.Layer):
    """
    Continuous-filter convolution layer
    """
    def __init__(self, fan_in, fan_out, n_filters,
                 activation=None):
        super().__init__()
        self._fan_in = fan_in
        self._fan_out = fan_out
        self._n_filters = n_filters
        self.activation = activation


        self.in2fac = tf.keras.layers.Dense(self._n_filters, use_bias=False)
        self.fac2out = tf.keras.layers.Dense(self._fan_out, use_bias=True, activation=self.activation)
        self.pool = PoolSegments()

    def call(self, x, w, seg_i, idx_j):
        '''
        :param x (num_atoms, num_feats): input
        :param w (num_interactions, num_filters): filters
        :param seg_i (num_interactions,): segments of atom i
        :param idx_j: (num_interactions,): indices of atom j
        :return: convolution x * w
        '''
        # to filter-space
        f = self.in2fac(x)

        # filter-wise convolution

        f = tf.gather(f, idx_j, batch_dims=1)

        wf = w * f

        conv = self.pool(wf, seg_i)

        # to output-space
        y = self.fac2out(conv)
        return y

