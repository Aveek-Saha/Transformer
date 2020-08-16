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

    return pos_enc[np.newaxis, :]


# pos_encoding = positional_encoding(50, 512)
# print(pos_encoding.shape)


def scaled_dot_product_attention(q, k, v, mask=None):
    """Scaled dot product attention"""

    d_k = tf.shape(k)[-1]
    # k_t = tf.transpose(k)

    attn = tf.linalg.matmul(q, k, transpose_b=True)
    scaled_attn = attn/np.sqrt(d_k)

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


def multi_head_attention(q, k, v, num_heads=8, mask=None):

    d_model = tf.shape(q)[-1]
    batch_size = tf.shape(q)[0]
    seq_len_1 = tf.shape(q)[1]
    seq_len_2 = tf.shape(k)[1]

    q = tf.keras.layers.Dense(d_model)(q)
    k = tf.keras.layers.Dense(d_model)(k)
    v = tf.keras.layers.Dense(d_model)(v)

    q = tf.concat(tf.split(q, num_heads, axis=2), axis=0)
    k = tf.concat(tf.split(k, num_heads, axis=2), axis=0)
    v = tf.concat(tf.split(v, num_heads, axis=2), axis=0)

    scaled_attn, scaled_attn_weights = scaled_dot_product_attention(q, k, v)
    scaled_attn_weights = tf.reshape(
        scaled_attn_weights, (batch_size, num_heads, seq_len_1, seq_len_2))

    scaled_attn = tf.concat(tf.split(scaled_attn, num_heads, axis=0), axis=2)

    output = tf.keras.layers.Dense(d_model)(scaled_attn)

    return output, scaled_attn_weights


# y = tf.random.uniform((2, 60, 512))  # (batch_size, encoder_sequence, d_model)
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

print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)

def DecoderLayer(x, encoder_out, d_model, d_ff, num_heads=8, rate=0.1, training=False, padding_mask=None, look_ahead_mask=None):
    scaled_attn, scaled_attn_weights = multi_head_attention(x, x, x, num_heads)
    scaled_attn = tf.keras.layers.Dropout(rate)(scaled_attn, training=training)
    output1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(x + scaled_attn)

    scaled_attn2, scaled_attn_weights2 = multi_head_attention(
        output1, encoder_out, encoder_out, num_heads)
    scaled_attn2 = tf.keras.layers.Dropout(
        rate)(scaled_attn2, training=training)
    output2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(x + scaled_attn2)

    ff_output = pointwise_feed_forward(output2, (d_ff, d_model))
    ff_output = tf.keras.layers.Dropout(rate)(ff_output, training=training)
    output3 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6)(output2 + ff_output)

    return output3, scaled_attn_weights, scaled_attn_weights2


# sample_decoder_layer_output, _, _ = DecoderLayer(
#     tf.random.uniform((64, 50, 512)), sample_encoder_layer_output, 512, 2048)

# print(sample_decoder_layer_output.shape)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.input_vocab_size = input_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

    def call(self, x, training=False, mask=None):

        pos_enc = positional_encoding(
            self.maximum_position_encoding, self.d_model)

        seq_len = tf.shape(x)[1]

        x = tf.keras.layers.Embedding(self.input_vocab_size, self.d_model)(x)
        x *= np.sqrt(self.d_model)
        x += pos_enc[:, :seq_len, :]

        x = tf.keras.layers.Dropout(self.rate)(x, training=training)

        for i in range(self.num_layers):
            x = EncoderLayer(x, self.d_model, self.d_ff,
                             self.num_heads, self.rate, training, mask)

        return x


sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                         d_ff=2048, input_vocab_size=8500,
                         maximum_position_encoding=10000)
temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

# print(sample_encoder.trainable_variables)  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.target_vocab_size = target_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

    def call(self, x, enc_output, training=None, look_ahead_mask=None, padding_mask=None):
        pos_enc = positional_encoding(
            self.maximum_position_encoding, self.d_model)

        seq_len = tf.shape(x)[1]
        attn_weights = []

        x = tf.keras.layers.Embedding(self.target_vocab_size, self.d_model)(x)
        x *= np.sqrt(self.d_model)
        x += pos_enc[:, :seq_len, :]

        x = tf.keras.layers.Dropout(self.rate)(x, training=training)

        for i in range(self.num_layers):
            x, weight_block_1, weight_block_2 = DecoderLayer(x, enc_output, self.d_model, self.d_ff, self.num_heads,
                                                             self.rate, training, look_ahead_mask, padding_mask)
            attn_weights.append({
                "block_1": weight_block_1,
                "block_2": weight_block_2
            })

        return x, attn_weights


sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                         d_ff=2048, target_vocab_size=8000,
                         maximum_position_encoding=5000)
temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

output, attn = sample_decoder(temp_input,
                              enc_output=sample_encoder_output,
                              training=False,
                              look_ahead_mask=None,
                              padding_mask=None)

print(output.shape, attn[1]['block_2'].shape)
