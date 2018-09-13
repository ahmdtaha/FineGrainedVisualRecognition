import tensorflow as tf
import configuration as config

def poly_lr(global_step):
    starter_learning_rate = config.learning_rate
    end_learning_rate = config.end_learning_rate
    decay_steps = config.train_iters

    learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                              decay_steps, end_learning_rate,
                                              power=1)
    return learning_rate