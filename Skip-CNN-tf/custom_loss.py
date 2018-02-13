import tensorflow as tf

def cossim(a, b):
    dot = tf.cast(tf.tensordot(a, b, axes=1), tf.float32)

    norm1 = tf.sqrt(tf.cast(tf.tensordot(a, a, axes=1), tf.float32))
    norm2 = tf.sqrt(tf.cast(tf.tensordot(b, b, axes=1), tf.float32))

    mycossi = tf.div(dot, tf.multiply(norm1, norm2))

    return mycossi
