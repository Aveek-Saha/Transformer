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


def scaled_dot_product_attention(q, k, v, mask=None):
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


def EncoderLayer(x, d_model, d_ff, num_heads=8, rate=0.1, training=False, mask=None):

    scaled_attn, scaled_attn_weights = multi_head_attention(
        x, x, x, num_heads, mask)
    scaled_attn = tf.keras.layers.Dropout(rate)(scaled_attn, training=training)
    output1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + scaled_attn)

    ff_output = pointwise_feed_forward(output1, (d_ff, d_model))
    ff_output = tf.keras.layers.Dropout(rate)(ff_output, training=training)
    output2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(output1 + ff_output)

    return output2


sample_encoder_layer_output = EncoderLayer(
    tf.random.uniform((64, 43, 512)), 512, 2048)

# print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)

def DecoderLayer(x, encoder_out, d_model, d_ff, num_heads=8, rate=0.1, training=False, padding_mask=None, look_ahead_mask=None):
    scaled_attn, scaled_attn_weights = multi_head_attention(x, x, x, num_heads)
    scaled_attn = tf.keras.layers.Dropout(rate)(scaled_attn, training=training)
    output1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(x + scaled_attn)

    scaled_attn2, scaled_attn_weights2 = multi_head_attention(
        output1, encoder_out, encoder_out, num_heads)
    scaled_attn2 = tf.keras.layers.Dropout(rate)(scaled_attn2, training=training)
    output2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(x + scaled_attn2)

    ff_output = pointwise_feed_forward(output2, (d_ff, d_model))
    ff_output = tf.keras.layers.Dropout(rate)(ff_output, training=training)
    output3 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(output2 + ff_output)

    return output3, scaled_attn_weights, scaled_attn_weights2


sample_decoder_layer_output, _, _ = DecoderLayer(
    tf.random.uniform((64, 50, 512)), sample_encoder_layer_output, 512, 2048)

print(sample_decoder_layer_output.shape)

