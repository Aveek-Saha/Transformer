import tensorflow as tf
import numpy as np
import math

# print(tf.__version__)


def positional_encoding(pos, d_model):
    pos_enc = np.array([
        [p / np.power(10000, (i-(i % 2))/d_model)
         for i in range(d_model)]
        for p in range(pos)])

    pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
    pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])

    return tf.cast(pos_enc[np.newaxis, :], tf.float32)


def create_padding_mask(seq):
  mask = tf.cast(tf.math.equal(seq, 0), tf.float32)

  return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask


def scaled_dot_product_attention(q, k, v, mask=None):

    d_k = tf.cast(tf.shape(k)[-1], tf.float32)

    attn = tf.linalg.matmul(q, k, transpose_b=True)
    scaled_attn = attn/tf.math.sqrt(d_k)

    if mask is not None:
        scaled_attn += (mask * -1e9)

    attn_weights = tf.nn.softmax(scaled_attn, axis=-1)

    output = tf.linalg.matmul(attn_weights, v)

    return output, attn_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads=8):
        super(MultiHeadAttention, self).__init__()

        self.dense_q = tf.keras.layers.Dense(d_model)
        self.dense_k = tf.keras.layers.Dense(d_model)
        self.dense_v = tf.keras.layers.Dense(d_model)

        self.dense_out = tf.keras.layers.Dense(d_model)

        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

    def call(self, q, k, v, mask=None):

        batch_size = tf.shape(q)[0]
        seq_len_q = tf.shape(q)[1]
        seq_len_k = tf.shape(k)[1]
        seq_len_v = tf.shape(v)[1]

        q = self.dense_q(q)
        k = self.dense_k(k)
        v = self.dense_v(v)

        q = tf.reshape(tf.concat(tf.split(q, self.num_heads, axis=2),
                                 axis=0), (batch_size, -1, seq_len_q, self.depth))
        k = tf.reshape(tf.concat(tf.split(k, self.num_heads, axis=2),
                                 axis=0), (batch_size, -1, seq_len_k, self.depth))
        v = tf.reshape(tf.concat(tf.split(v, self.num_heads, axis=2),
                                 axis=0), (batch_size, -1, seq_len_v, self.depth))

        scaled_attn, scaled_attn_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attn = tf.reshape(
            scaled_attn, (batch_size, -1, self.d_model))

        output = self.dense_out(scaled_attn)

        return output, scaled_attn_weights


def pointwise_feed_forward(d_ff, d_model):

    output = tf.keras.Sequential([
        tf.keras.layers.Dense(d_ff, 'relu'),
        tf.keras.layers.Dense(d_model)
    ])

    return output


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, num_heads=8, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff = pointwise_feed_forward(d_ff, d_model)
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)

    def call(self, x, training=False, mask=None):

        scaled_attn, scaled_attn_weights = self.multi_head_attention(
            x, x, x, mask)
        scaled_attn = self.dropout1(scaled_attn, training=training)
        output1 = self.ln1(x + scaled_attn)

        ff_output = self.ff(output1)
        ff_output = self.dropout2(ff_output, training=training)
        output2 = self.ln2(output1 + ff_output)

        return output2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, num_heads=8, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ff = pointwise_feed_forward(d_ff, d_model)

        self.multi_head_attention1 = MultiHeadAttention(d_model, num_heads)
        self.multi_head_attention2 = MultiHeadAttention(d_model, num_heads)

    def call(self, x, encoder_out, training=False, padding_mask=None, look_ahead_mask=None):

        scaled_attn, scaled_attn_weights = self.multi_head_attention1(
            x, x, x, look_ahead_mask)
        scaled_attn = self.dropout1(scaled_attn, training=training)
        output1 = self.ln1(x + scaled_attn)

        scaled_attn2, scaled_attn_weights2 = self.multi_head_attention2(
            output1, encoder_out, encoder_out, padding_mask)
        scaled_attn2 = self.dropout2(scaled_attn2, training=training)
        output2 = self.ln2(x + scaled_attn2)

        ff_output = self.ff(output2)
        ff_output = self.dropout3(ff_output, training=training)
        output3 = self.ln3(output2 + ff_output)

        return output3, scaled_attn_weights, scaled_attn_weights2
