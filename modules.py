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


# pos_encoding = positional_encoding(50, 512)
# print(pos_encoding.shape)

def create_padding_mask(seq):
  mask = tf.cast(tf.math.equal(seq, 0), tf.float32)

  return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask


def scaled_dot_product_attention(q, k, v, mask=None):
    """Scaled dot product attention"""

    d_k = tf.cast(tf.shape(k)[-1], tf.float32)
    # k_t = tf.transpose(k)

    attn = tf.linalg.matmul(q, k, transpose_b=True)
    scaled_attn = attn/tf.math.sqrt(d_k)

    if mask is not None:
        scaled_attn += (mask * -1e9)

    attn_weights = tf.nn.softmax(scaled_attn, axis=-1)

    output = tf.linalg.matmul(attn_weights, v)

    return output, attn_weights


# k = tf.constant([[10, 0, 0],
#                  [0, 10, 0],
#                  [0, 0, 10],
#                  [0, 0, 10]], dtype=tf.float32)
# v = tf.constant([[1, 0],
#                  [10, 0],
#                  [100, 5],
#                  [1000, 6]], dtype=tf.float32)
# q = tf.constant([[0, 0, 10],
#                  [0, 10, 0],
#                  [10, 10, 0]], dtype=tf.float32)

# out, attn = scaled_dot_product_attention(q, k, v, None)
# print(attn)
# print(out)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads=8):
        super(MultiHeadAttention, self).__init__()

        self.dense_q = tf.keras.layers.Dense(d_model)
        self.dense_k = tf.keras.layers.Dense(d_model)
        self.dense_v = tf.keras.layers.Dense(d_model)

        self.dense_out = tf.keras.layers.Dense(d_model)

        self.num_heads = num_heads

    def call(self, q, k, v, mask=None):

        d_model = q.shape.as_list()[-1]
        batch_size = tf.shape(q)[0]
        seq_len_1 = tf.shape(q)[1]
        seq_len_2 = tf.shape(k)[1]

        q = self.dense_q(q)
        k = self.dense_k(k)
        v = self.dense_v(v)

        q = tf.concat(tf.split(q, num_heads, axis=2), axis=0)
        k = tf.concat(tf.split(k, num_heads, axis=2), axis=0)
        v = tf.concat(tf.split(v, num_heads, axis=2), axis=0)

        scaled_attn, scaled_attn_weights = scaled_dot_product_attention(
            q, k, v, mask)
        scaled_attn_weights = tf.reshape(
            scaled_attn_weights, (batch_size, self.num_heads, seq_len_1, seq_len_2))

        scaled_attn = tf.concat(
            tf.split(scaled_attn, self.num_heads, axis=0), axis=2)

        output = self.dense_out(scaled_attn)

        return output, scaled_attn_weights


# y = tf.random.uniform((2, 60, 512))  # (batch_size, encoder_sequence, d_model)
# out, attn = multi_head_attention(y, y, y)
# print(out.shape, attn.shape)


def pointwise_feed_forward(d_ff, d_model):

    output = tf.keras.Sequential([
        tf.keras.layers.Dense(d_ff, 'relu'),
        tf.keras.layers.Dense(d_model)
    ])

    return output


# res = pointwise_feed_forward(tf.random.uniform((64, 50, 512)), (2048, 512))
# print(res.shape)


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

    scaled_attn, scaled_attn_weights = self.multi_head_attention(x, x, x, mask)
    scaled_attn = self.dropout1(scaled_attn, training=training)
    output1 = self.ln1(x + scaled_attn)

    ff_output = self.ff(output1)
    ff_output = self.dropout2(ff_output, training=training)
    output2 = self.ln2(output1 + ff_output)

    return output2


# sample_encoder_layer_output = EncoderLayer(
#     tf.random.uniform((64, 43, 512)), 512, 2048)

# print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)

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
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        
    def call(self, x, encoder_out, training=False, padding_mask=None, look_ahead_mask=None):

        scaled_attn, scaled_attn_weights = self.multi_head_attention(x, x, x, mask)
        scaled_attn = self.dropout1(scaled_attn, training=training)
        output1 = self.ln1(x + scaled_attn)

        scaled_attn2, scaled_attn_weights2 = self.multi_head_attention(
            output1, encoder_out, encoder_out)
        scaled_attn2 = self.dropout2(scaled_attn2, training=training)
        output2 = self.ln2(x + scaled_attn2)

        ff_output = self.ff(output2)
        ff_output = self.dropout3(ff_output, training=training)
        output3 = self.ln3(output2 + ff_output)

        return output3, scaled_attn_weights, scaled_attn_weights2


# sample_decoder_layer_output, _, _ = DecoderLayer(
#     tf.random.uniform((64, 50, 512)), sample_encoder_layer_output, 512, 2048)

# print(sample_decoder_layer_output.shape)
