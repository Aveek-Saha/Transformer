import tensorflow as tf
import numpy as np

from modules import positional_encoding, scaled_dot_product_attention, MultiHeadAttention, pointwise_feed_forward
from modules import EncoderLayer, DecoderLayer


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.pos_enc = positional_encoding(maximum_position_encoding, d_model)

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.dropout = tf.keras.layers.Dropout(rate)

        self.encoder_layers = [EncoderLayer(
            d_model, d_ff, num_heads, rate) for x in range(num_layers)]

    def call(self, x, training=False, mask=None):

        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= np.sqrt(self.d_model)
        x += self.pos_enc[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.pos_enc = positional_encoding(maximum_position_encoding, d_model)

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.dropout = tf.keras.layers.Dropout(rate)

        self.decoder_layers = [DecoderLayer(
            d_model, d_ff, num_heads, rate) for x in range(num_layers)]

    def call(self, x, enc_output, training=None, look_ahead_mask=None, padding_mask=None):

        seq_len = tf.shape(x)[1]
        attn_weights = []

        x = self.embedding(x)
        x *= np.sqrt(self.d_model)
        x += self.pos_enc[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, weight_block_1, weight_block_2 = self.decoder_layers[i](
                x, enc_output, training, padding_mask, look_ahead_mask)
            attn_weights.append({
                "block_1": weight_block_1,
                "block_2": weight_block_2
            })

        return x, attn_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size,
                 target_vocab_size, pos_enc_input, pos_enc_target, rate=0.1):

        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff,
                               input_vocab_size,  pos_enc_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, target_vocab_size,
                               pos_enc_target, rate)

        self.dense = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, targets, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

        encoder_out = self.encoder(inputs, training, enc_padding_mask)

        decoder_out, attn_weights = self.decoder(
            targets, encoder_out, training, look_ahead_mask, dec_padding_mask)

        output = self.dense(decoder_out)

        return output, attn_weights
