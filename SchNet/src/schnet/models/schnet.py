import schnet.nn.layers as L
import tensorflow as tf
from schnet.nn.activation import shifted_softplus



class SchNetFilter(tf.keras.layers.Layer):
    def __init__(self, n_in, n_filters):
        super().__init__()
        self.n_in = n_in
        self.n_filters = n_filters
        self.dense1 = tf.keras.layers.Dense(self.n_filters, activation=shifted_softplus)
        self.dense2 = tf.keras.layers.Dense(self.n_filters, activation=shifted_softplus)
        self.pooling = L.PoolSegments()

    def call(self, dijk, seg_j):
        h = self.dense1(dijk)
        w_ijk = self.dense2(h)
        w_ij = self.pooling(w_ijk, seg_j)
        return w_ij



class SchNetInteractionBlock(tf.keras.layers.Layer):
    def __init__(self, n_in, n_basis, n_filters, pool_mode='sum'):
        super().__init__()
        self.n_in = n_in
        self.n_basis = n_basis
        self.n_filters = n_filters
        self.pool_mode = pool_mode

        self.filternet = SchNetFilter(self.n_in, self.n_filters)
        self.cfconv = L.CFConv(self.n_basis, self.n_basis, self.n_filters,activation=shifted_softplus)
        self.dense =tf.keras.layers.Dense(self.n_basis)

    def call(self, x, dijk, idx_j, seg_i, seg_j, ratio_j=None):
        w = self.filternet(dijk, seg_j)
        h = self.cfconv(x, w, seg_i, idx_j)
        v = self.dense(h)
        y = x + v
        return y, v

    def _calc_filter(self, dijk, seg_j, ratio_j):
        w = self.filternet(dijk, seg_j, ratio_j)
        return w


class SchNet(tf.keras.models.Model):
    def __init__(self, n_interactions, n_basis, n_filters, cutoff,
                 gap=0.1,
                 filter_pool_mode='sum',
                 return_features=False,
                 shared_interactions=False,
                 atomization_energy=False,
                 n_embeddings=100):
        super().__init__()
        self.n_interactions = n_interactions
        self.n_basis = n_basis
        self.n_filters = n_filters
        self.n_embeddings = n_embeddings
        self.cutoff = cutoff
        self.atomization_energy = atomization_energy
        self.shared_interactions = shared_interactions
        self.return_features = return_features
        self.filter_pool_mode = filter_pool_mode
        self.gap = gap

        self.atom_embedding = L.Embedding(self.n_embeddings, self.n_basis)

        self.dist = L.EuclideanDistances()
        self.rbf = L.RBFExpansion(0., self.cutoff, self.gap)

        if self.shared_interactions:
            self.interaction_blocks = \
                [
                    SchNetInteractionBlock(
                        self.rbf.fan_out, self.n_basis, self.n_filters,
                        pool_mode=self.filter_pool_mode)
                ] * self.n_interactions
        else:
            self.interaction_blocks = [
                SchNetInteractionBlock(
                    self.rbf.fan_out, self.n_basis, self.n_filters)
                for i in range(self.n_interactions)]

        self.dense1 = tf.keras.layers.Dense(self.n_basis // 2,
                              activation=shifted_softplus)
        self.dense2 =tf.keras.layers.Dense(1)
        self.atom_pool = L.PoolSegments('sum')


        self.e0 = L.Embedding(self.n_embeddings, 1)

    def call(self,input):

        seg_m =  input['seg_m']
        idx_ik =  input['idx_ik']
        seg_i =  input['seg_i']
        idx_j =   input['idx_j']
        idx_jk = input['idx_jk']
        seg_j =  input['seg_j']
        z=  input['numbers']
        r =  input['positions']
        offsets =  input['offset']
        ratio_j =  input['ratio_j']

        # embed atom species
        x = self.atom_embedding(z)

        # interaction features
        dijk = self.dist(r, offsets, idx_ik, idx_jk)
        dijk = self.rbf(dijk)

        # interaction blocks
        V = []
        for iblock in self.interaction_blocks:
            x, v = iblock(x, dijk, idx_j, seg_i, seg_j, ratio_j)
            # x = print_shape(x)
            V.append(v)

        # output network
        h = self.dense1(x)
        y_i = self.dense2(h)

        # scale energy contributions
        if self.e0 is not None and not self.atomization_energy:
            y_i += self.e0(z)

        y = self.atom_pool(y_i, seg_m)

        if not self.return_features:
            return y
        return y, y_i, x, V






def print_shape(t, name=None):
    if name is None:
        name = t.name
    return tf.Print(t, [tf.shape(t), t], summarize=20, message=name)
