import tensorflow as tf

def poly_lr(global_step,cfg):
    starter_learning_rate = cfg.learning_rate
    end_learning_rate = cfg.end_learning_rate
    decay_steps = cfg.train_iters

    learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                              decay_steps, end_learning_rate,
                                              power=1)
    return learning_rate