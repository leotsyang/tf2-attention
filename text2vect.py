
import tensorflow as tf
import tensorflow_text as tf_text



def tf_lower_and_split_punct(text):
    text = tf_text.normalize_utf8(text, 'NFKD')

    text = tf.strings.lower(text)

    # keep space , a-z, and some punctuations

    text = tf.strings.regex_replace(text, '[^ a-z.?!,]', '')

    # add space to both side of puct
    text = tf.strings.regex_replace(text, '[.?!,]', ' \0 ')

    # remove leading and trailing spaces
    text = tf.strings.strip(text)

    # add start of sentence and end of sentences token
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


class Text2Vect(tf.keras.layers.Layer):

    def __init__(self, vocab_size, training_text):
        self.text_processor = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=vocab_size,
            standardize=tf_lower_and_split_punct
        )

        self.text_processor.adapt(training_text)

        vocabs = self.text_processor.get_vocabulary()
        print("VOCABS ...")
        print(vocabs[200:210])
        print("VOCABS ...")


    def __call__(self, input_text):
        return self.text_processor(input_text)



def text_processor(vocab_size):
    return tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=vocab_size,
                                                                        standardize=tf_lower_and_split_punct)




if __name__ == '__main__':

    txt = tf_lower_and_split_punct('Todava est en casa?**$#')
    print(txt)
