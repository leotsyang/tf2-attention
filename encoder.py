import tensorflow as tf
import numpy as np

from shape_checker import ShapeChecker
from text2vect import Text2Vect

class Encoder(tf.keras.layers.Layer):

    def __init__(self, input_vocab_size, embedding_dim, enc_units):
        '''

        :param input_vocab_size:
        :param embedding_dim:
        :param enc_units: the output units of GRU layer
        :return:
        '''

        super(Encoder, self).__init__()

        self.enc_units = enc_units
        self.input_vocab_size = input_vocab_size
        self.embedding_dim = embedding_dim

        # embedding layer convert the  text token to embedding vec
        self.embedding = tf.keras.layers.Embedding(self.input_vocab_size, embedding_dim)

        #  then GRU layer with output of enc_units
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def __call__(self, tokens, state=None):
        '''
        :param tokens: text to vector tokens
        :param state: initial cell state
        :return:
        '''

        shape_checker = ShapeChecker()

        # the following check essentially assigned token shape into shape_checker
        #  shape = {'batch' :  token.shape[0], 's': token.shape[1]}

        shape_checker(tokens, ('batch', 's'))

        # word embedding:
        vectors = self.embedding(tokens)

        # it checks first 2 dimension, then store embed_dim into checker itself.
        shape_checker(vectors, ('batch', 's', 'embed_dim'))

        enc_output, enc_state = self.gru(vectors,initial_state=state)

        shape_checker(enc_output, ('batch', 's', 'enc_units'))
        shape_checker(enc_state, ('batch', 'enc_units'))

        return enc_output, enc_state






