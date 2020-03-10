import tensorflow as tf



class EuclideanDistances(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, r, offsets, idx_ik, idx_jk):
        ri = tf.gather(r, idx_ik,batch_dims=1)
        rj = tf.gather(r, idx_jk,batch_dims=1) + offsets
        rij = ri - rj

        dij2 = tf.reduce_sum(rij ** 2, -1, keepdims=True)
        dij = tf.sqrt(tf.nn.relu(dij2))
        return dij
