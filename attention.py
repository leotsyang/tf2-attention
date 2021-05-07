import tensorflow as tf
from shape_checker import ShapeChecker


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()

        # add 2 FC layers for W1 and W2 in Bahdanau Attention
        # units here is actually 'attn_units' ?
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)

        self.attention = tf.keras.layers.AdditiveAttention()

    def __call__(self, query, value, mask):

        shape_checker = ShapeChecker()

        # query units , i.e. decoder RNN hidden output units
        # t, decoder time steps
        shape_checker(query, ('batch', 't', 'query_units'))

        # value_units, i.e. encoder RNN hidden state unit size
        # s, encoder time steps
        shape_checker(value, ('batch', 's', 'value_units'))

        shape_checker(mask, ('batch', 's'))

        w1_query = self.W1(query)
        shape_checker(w1_query, ('batch', 't', 'attn_units'))

        w2_key = self.W2(value)
        shape_checker(w2_key, ('batch', 's', 'attn_units'))

        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask

        context_vector, attention_weights = self.attention(
            inputs=[w1_query, value, w2_key],
            mask=[query_mask, value_mask],
            return_attention_scores=True,
        )
        shape_checker(context_vector, ('batch', 't', 'value_units'))
        shape_checker(attention_weights, ('batch', 't', 's'))

        return context_vector, attention_weights

