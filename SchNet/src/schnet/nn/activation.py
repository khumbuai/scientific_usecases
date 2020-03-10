import tensorflow as tf

def swish(x, beta=1.0):
    return x * tf.sigmoid(beta * x)

def shifted_softplus(x):
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (tf.Tensor): input tensor.

    Returns:
        tf.Tensor: shifted soft-plus of input.

    """
    return tf.keras.backend.log(0.5*tf.keras.backend.exp(x)+0.5)
