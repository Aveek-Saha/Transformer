import tensorflow as tf
import numpy as np

from modules import positional_encoding, scaled_dot_product_attention, multi_head_attention, pointwise_feed_forward
from modules import EncoderLayer, DecoderLayer


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


# sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
#                          d_ff=2048, input_vocab_size=8500,
#                          maximum_position_encoding=10000)
# temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

# sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

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


# sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
#                          d_ff=2048, target_vocab_size=8000,
#                          maximum_position_encoding=5000)
# temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

# output, attn = sample_decoder(temp_input,
#                               enc_output=sample_encoder_output,
#                               training=False,
#                               look_ahead_mask=None,
#                               padding_mask=None)

# print(output.shape, attn[1]['block_2'].shape)


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size,
                 target_vocab_size, pos_enc_input, pos_enc_target, rate=0.1):

        super(Transformer, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.pos_enc_input = pos_enc_input
        self.pos_enc_target = pos_enc_target
        self.rate = rate

    def call(self, inputs, targets, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

        encoder_out = Encoder(self.num_layers, self.d_model, self.num_heads, self.d_ff,
                              self.input_vocab_size, self. pos_enc_input, self.rate)(inputs, training, enc_padding_mask)

        decoder_out, attn_weights = Decoder(self.num_layers, self.d_model, self.num_heads, self.d_ff, self.target_vocab_size,
                                            self.pos_enc_target, self.rate)(targets, encoder_out, training, look_ahead_mask, dec_padding_mask)

        output = tf.keras.layers.Dense(self.target_vocab_size)(decoder_out)

        return output, attn_weights


# sample_transformer = Transformer(
#     num_layers=2, d_model=512, num_heads=8, d_ff=2048,
#     input_vocab_size=8500, target_vocab_size=8000,
#     pos_enc_input=10000, pos_enc_target=6000)

# temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
# temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

# fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
#                                enc_padding_mask=None,
#                                look_ahead_mask=None,
#                                dec_padding_mask=None)

# print(fn_out.shape)
