import tensorflow as tf
from shape_checker import ShapeChecker


class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self):
        self.name = 'masked_loss'
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def __call__(self, y_true, y_pred):
        '''

        :param y_true:
        :param y_pred: note y_pred is logit output
        :return:
        '''
        shape_checker = ShapeChecker()

        shape_checker(y_true, ('batch', 't'))
        shape_checker(y_pred, ('batch', 't', 'logits'))

        # calculate loss by spare_categorical_crossentropy
        loss = self.loss(y_true, y_pred)

        # mask off the losses on padding
        mask = tf.cast(y_true != 0, tf.float32)
        shape_checker(mask, ('batch', 't'))

        # elementwise multiplication to mask out.
        loss *= mask

        return tf.reduce_sum(loss)
