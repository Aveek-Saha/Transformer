import tensorflow as tf
import numpy as np
import math

print(tf.__version__)


def scaled_dot_product_attention(q, k, v, mask = None):
    """Scaled dot product attention"""

    d_k = tf.shape(k)[-1]
    k_t = tf.transpose(k)
    
    attn = tf.linalg.matmul(q, k_t)
    scaled_attn = attn/math.sqrt(d_k)

    attn_weights = tf.nn.softmax(scaled_attn)

    output = tf.linalg.matmul(attn_weights, v)

    return output, attn_weights






