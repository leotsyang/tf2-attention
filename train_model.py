from abc import ABC

import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
from attention import BahdanauAttention
from shape_checker import ShapeChecker
from decoder import DecoderInput, DecoderOutput


class TrainTranslator(tf.keras.Model):
    def __init__(self, embedding_dim, units, input_text_processor, output_text_processor,
                 use_tf_function=True):
        super().__init__()

        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        self.use_tf_function = use_tf_function
        self.shape_checker = ShapeChecker()

        self.encoder = Encoder(len(input_text_processor.get_vocabulary()),
                               embedding_dim,
                               units)

        self.decoder = Decoder(len(output_text_processor.get_vocabulary()),
                               embedding_dim,
                               units)

    def _tf_train_step(self, inputs):
        pass


    def _preprocess(self, input_text, target_text):

        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)

        self.shape_checker(input_tokens, ('batch', 's'))
        self.shape_checker(target_tokens, ('batch', 't'))

        # convert IDs TO masks
        # ? what is the purpose of masks
        input_mask = input_tokens != 0

        target_mask = target_tokens != 0

        return input_tokens, input_mask, target_tokens, target_mask

    def _loop_step(self, new_tokens, input_mask, enc_output, dec_state):
        """
        executes the decoder and calculates the incremental loss and new decoder state (dec_state).
        :param new_token:
        :param input_mask:
        :param enc_output:
        :param dec_state:
        :return:
        """

        # when use target text to train the model, next word is the prediction for input of current word.
        decoder_input_token, decoder_pred_token = new_tokens[:, 0:1], new_tokens[:, 1:2]

        decoder_input = DecoderInput(new_tokens=decoder_input_token, enc_output=enc_output, mask=input_mask)

        dec_result, dec_state = self.decoder(decoder_input, state=dec_state)

        self.shape_checker(dec_result.logits, ('batch', 't1', 'logits'))
        self.shape_checker(dec_result.attention_weights, ('batch', 't1', 's'))
        self.shape_checker(dec_state, ('batch', 'dec_units'))

        y = decoder_pred_token
        y_pred = dec_result.logits

        step_loss = self.loss(y, y_pred)

        return step_loss, dec_state

    def _train_step(self, inputs):
        '''

        :param inputs: batch input/target from tf.dataset
        :return:
        '''
        input_text, target_text = inputs

        # preprocess into token and masks
        input_tokens, input_mask, target_tokens, target_mask = self._preprocess(input_text, target_text)

        # target tokens of shape (batch, t)
        max_target_length = tf.shape(target_tokens)[1]

        # gradient calculation
        with tf.GradientTape() as tape:

            # encoding for ALL timesteps in one go
            enc_output, enc_state = self.encoder(input_tokens)

            self.shape_checker(enc_output, ('batch', 's', 'enc_units'))
            self.shape_checker(enc_state, ('batch', 'enc_units'))

            # decoding , time step by time step
            dec_state = enc_state
            loss = tf.constant(0.0)

            # -1 as we pass 2 tokens in each iteration
            for t in tf.range(max_target_length - 1):

                new_tokens = target_tokens[:, t:t+2]
                step_loss, dec_state = self._loop_step(new_tokens, input_mask, enc_output, dec_state)

                loss += step_loss
            average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))


        # apply an optimization step with gradient descent
        variables = self.trainable_variables
        gradients = tape.gradient(average_loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))

        return {'batch_loss': average_loss}

    def train_step(self, inputs):

        if self.use_tf_function:
            return self._tf_train_step(inputs)
        else:
            return self._train_step(inputs)





