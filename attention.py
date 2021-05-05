import tensorflow as tf
import numpy as np
import urllib
from pathlib import Path
import tensorflow_text as tf_text
import typing
from typing import Any, Tuple

from tensorflow.keras.layers.experimental import preprocessing
from text2vect import Text2Vect
from shape_checker import ShapeChecker

from encoder import Encoder


def load_anki_data(file_path):

    path_data_dir = Path('.')

    path_nmt_file = path_data_dir / 'data' / file_path
    fulltext = path_nmt_file.read_text(encoding='utf-8')

    lines = fulltext.splitlines()
    pairs = [line.split('\t') for line in lines]

    txt_in = [inp for targ, inp, _ in pairs]
    txt_out = [targ for targ, inp, _ in pairs]

    return txt_out, txt_in


def prepare_dataset(_input, _target):

    BUFFER_SIZE = len(_input)
    BATCH_SIZE = 64

    dataset = tf.data.Dataset.from_tensor_slices((_input, _target))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    type(dataset)
    print("type of the dataset {}".format(type(dataset)))



    return dataset

def tf_lower_and_split_punct(text):
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)

    # add start of sentence and end of sentences token
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')

    print(text)

    return text


def encode():

    max_vocab_size = 50000
    embedding_dim = 256
    encoder_units = 1024

    _target, _input = load_anki_data('spa.txt')


    # train text2vec layer for both input and target text
    input_text_processor = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_vocab_size,
            standardize=tf_lower_and_split_punct
        )
    input_text_processor.adapt(_input)

    input_vocab_size = len(input_text_processor.get_vocabulary())


    output_text_processor = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=max_vocab_size,
        standardize=tf_lower_and_split_punct
    )

    output_text_processor.adapt(_target)
    output_vocab_size = len(output_text_processor.get_vocabulary())


    # construct encoder
    encoder = Encoder(input_vocab_size,
                       embedding_dim,
                       encoder_units)

    # prepare data as tf.dataset
    dataset = prepare_dataset(_input, _target)
    for batch_input, batch_target in dataset.take(1):
        print(batch_input[:5])


    # text to token
    batch_input_token = input_text_processor(batch_input)

    batch_enc_output, batch_enc_state = encoder(batch_input_token)

    print("batch input shape (batch): {}".format(batch_input.shape))
    print("batch token shape (batch,s): {}".format(batch_input_token.shape))
    print("batch encoder output (batch,s,enc_units): {}".format(batch_enc_output.shape))
    print("batch encoder state (batch,enc_units): {}".format(batch_enc_state.shape))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    encode()


