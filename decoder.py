'''
The decoder's job is to generate predictions for the next output token.

1. The decoder receives the complete encoder output.
2. It uses an RNN to keep track of what it has generated so far.
3. It uses its RNN output as the query to the attention over the encoder's output, producing the context vector.
4. It combines the RNN output and the context vector using Equation 3 (below) to generate the "attention vector".
5. It generates logit predictions for the next token based on the "attention vector".
'''

from typing import Any, Tuple, NamedTuple

import tensorflow as tf
from attention import BahdanauAttention
from shape_checker import ShapeChecker


class DecoderInput(NamedTuple):
    new_tokens: Any
    enc_output: Any
    mask: Any


class DecoderOutput(NamedTuple):
    logits: Any
    attention_weights: Any


class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super().__init__()

        self.dec_units = dec_units
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim

        # embeddign layer
        self.embedding = tf.keras.layers.Embedding(self.output_vocab_size, self.embedding_dim)

        # RNN layer,  use GRN in this example
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        # attention layer, use Bahdanau Attention:
        # note here we make  attention layer same output size as decode units
        self.attention = BahdanauAttention(self.dec_units)

        # FC layer to generate attention vector
        self.Wc = tf.keras.layers.Dense(dec_units, activation='tanh', use_bias=False)

        # followed by FC layer produce logits output, note here output size is the vocab size , different to encoder.
        self.fc = tf.keras.layers.Dense(self.output_vocab_size)

    def __call__(self, inputs: DecoderInput, state=None) -> Tuple[DecoderOutput, tf.Tensor]:
        shape_checker = ShapeChecker()

        shape_checker(inputs.new_tokens, ('batch', 't'))
        shape_checker(inputs.enc_output, ('batch', 's', 'enc_units'))
        shape_checker(inputs.mask, ('batch', 's'))

        if state is not None:
            shape_checker(state, ('batch', 'dec_units'))

        # step 1:  embedding:
        embedding_vec = self.embedding(inputs.new_tokens)
        shape_checker(embedding_vec, ('batch', 't', 'embedding_dim'))

        # step 2: RNN - GRU

        rnn_output, state = self.gru(embedding_vec,initial_state=state)

        shape_checker(rnn_output, ('batch', 't', 'dec_units'))
        # state to keep its shape after RNN
        shape_checker(state, ('batch', 'dec_units'))

        # step 3: do attention :
        context_vec, attention_weight = self.attention(query=rnn_output,
                                                       value=inputs.enc_output,
                                                       mask=inputs.mask)

        shape_checker(context_vec, ('batch', 't', 'dec_units'))
        shape_checker(attention_weight, ('batch', 't', 's'))

        # step 4:  join context_vec and rnn_output into attention_vec
        attention_vector = self.Wc(tf.concat([context_vec, rnn_output], axis=-1))
        shape_checker(attention_vector, ('batch', 't', 'dec_units'))

        # step 5:  FC layer for logit outputs
        logits = self.fc(attention_vector)
        shape_checker(logits, ('batch', 't', 'output_vocab_size'))

        return DecoderOutput(logits, attention_weight), state




