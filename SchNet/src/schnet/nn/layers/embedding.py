import numpy as np
import tensorflow as tf


class Embedding(tf.keras.layers.Layer):
    def __init__(self, n_embeddings, dim,
                 embedding_init=None):
        super().__init__()
        self._n_embeddings = n_embeddings
        self._dim = dim
        self._embedding_init = embedding_init



        self.embeddings = tf.keras.layers.Dense(self._dim, use_bias=False, kernel_initializer=self._embedding_init)

    def call(self, indices):
        I = np.eye(self._n_embeddings).astype(np.float32)
        ind = tf.nn.embedding_lookup(I, indices)
        y = self.embeddings(ind)
        return y
