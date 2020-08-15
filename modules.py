import tensorflow as tf
import numpy as np
import math

print(tf.__version__)

def positional_encoding(pos, d_model):
    pos_enc = np.array([
        [p / np.power(10000, (i-(i % 2))/d_model)
        for i in range(d_model)] 
        for p in range(pos)])

    pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
    pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])

    return(pos_enc)


# pos_encoding = positional_encoding(50, 512)
# print(pos_encoding.shape)


def scaled_dot_product_attention(q, k, v, mask = None):
    """Scaled dot product attention"""

    d_k = tf.shape(k)[-1]
    k_t = tf.transpose(k, [0, 2, 1])
    
    attn = tf.linalg.matmul(q, k_t)
    scaled_attn = attn/math.sqrt(d_k)

    attn_weights = tf.nn.softmax(scaled_attn)

    output = tf.linalg.matmul(attn_weights, v)

    return output, attn_weights


def multi_head_attention(q, k, v, num_heads=8, mask=None):

    d_model = tf.shape(q)[-1]

    q = tf.keras.layers.Dense(d_model)(q)
    k = tf.keras.layers.Dense(d_model)(k)
    v = tf.keras.layers.Dense(d_model)(v)

    q = tf.concat(tf.split(q, num_heads, axis=2), axis=0)
    k = tf.concat(tf.split(k, num_heads, axis=2), axis=0)
    v = tf.concat(tf.split(v, num_heads, axis=2), axis=0)

    scaled_attn, scaled_attn_weights = scaled_dot_product_attention(q, k, v)

    scaled_attn = tf.concat(tf.split(scaled_attn, num_heads, axis=0), axis=2)

    output = tf.keras.layers.Dense(d_model)(scaled_attn)

    return output, scaled_attn_weights


# y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
# out, attn = multi_head_attention(y, y, y)
# print(out.shape, attn.shape)


def pointwise_feed_forward(inputs, dim):

    output = tf.keras.layers.Dense(dim[0], 'relu')(inputs)
    output = tf.keras.layers.Dense(dim[1])(output)

    return output


# res = pointwise_feed_forward(tf.random.uniform((64, 50, 512)), (2048, 512))
# print(res.shape)



