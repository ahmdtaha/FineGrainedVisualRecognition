import tensorflow as tf
import constants as const
import configuration as config
import os
import numpy as np
from datetime import datetime
from pydoc import locate
import time
from data.tf_tuple_loader import TensorflowTupleLoader
from utils import tf_utils


def mprint(msg):
    print(msg)
    # logger.info(msg) # Use your own logger

def main():

    img_generator_class = locate(config.db_tuple_loader)
    args = dict()
    args['csv_file'] = config.train_csv_file
    args['img_size'] = const.max_frame_size
    args['gen_hot_vector'] = True
    train_iter = img_generator_class(args)
    args['batch_size'] = config.batch_size * 2
    args['csv_file'] = config.test_csv_file
    val_iter = img_generator_class(args)

    train_imgs, train_lbls = train_iter.imgs_and_lbls()
    val_imgs, val_lbls = val_iter.imgs_and_lbls()

    save_model_dir = config.model_save_path

    with tf.Graph().as_default():

        train_dataset = TensorflowTupleLoader(train_imgs, train_lbls, is_training=True).dataset
        val_dataset = TensorflowTupleLoader(val_imgs, val_lbls, is_training=False, batch_size=config.batch_size,
                                       repeat=False).dataset

        handle = tf.placeholder(tf.string, shape=[])

        iterator = tf.data.Iterator.from_string_handle(
            handle, train_dataset.output_types, train_dataset.output_shapes)
        images_ph, lbls_ph = iterator.get_next()

        training_iterator = train_dataset.make_one_shot_iterator()
        validation_iterator = val_dataset.make_initializable_iterator()

        network_class = locate(config.network_name)
        model = network_class(num_classes=config.num_classes, is_training=True, images_ph=images_ph, lbls_ph=lbls_ph)


        trainable_vars = tf.trainable_variables()
        if config.caffe_iter_size > 1:  ## Accumulated Gradient
            ## Creation of a list of variables with the same shape as the trainable ones
            # initialized with 0s
            accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in trainable_vars]
            zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            global_step = tf.Variable(0, name='global_step', trainable=False)
            learning_rate = tf_utils.poly_lr(global_step)
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)

            if config.caffe_iter_size > 1:  ## Accumulated Gradient

                grads = optimizer.compute_gradients(model.train_loss, trainable_vars)
                # Adds to each element from the list you initialized earlier with zeros its gradient (works because accum_vars and gvs are in the same order)
                accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(grads)]
                iter_size = config.caffe_iter_size
                # Define the training step (part with variable value update)
                train_op = optimizer.apply_gradients([(accum_vars[i] / iter_size, gv[1]) for i, gv in enumerate(grads)],
                                                     global_step=global_step)

            else:
                grads = optimizer.compute_gradients(model.train_loss)
                train_op = optimizer.apply_gradients(grads, global_step=global_step)

        # logger.info('=========================================================')
        # for v in tf.trainable_variables():
        #     mprint('trainable_variables:  {0} \t {1}'.format(str(v.name),str(v.shape)))


        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())


        now = datetime.now()
        if (config.tensorbaord_file == None):
            tb_path = config.tensorbaord_dir + now.strftime("%Y%m%d-%H%M%S")
        else:
            tb_path = config.tensorbaord_dir + config.tensorbaord_file

        start_iter = 0 # No Resume in this code version

        train_writer = tf.summary.FileWriter(tb_path, sess.graph)

        saver = tf.train.Saver()  # saves variables learned during training

        ckpt_file = os.path.join(save_model_dir, config.model_save_name)
        print('Model Path ', ckpt_file)



        load_model_msg = model.load_model(save_model_dir, ckpt_file, sess, saver, is_finetuning=True)
        mprint(load_model_msg)


        val_loss = tf.summary.scalar('Val_Loss', model.val_loss)
        val_acc_op = tf.summary.scalar('Batch_Val_Acc', model.val_accuracy)
        model_acc_op = tf.summary.scalar('Split_Val_Accuracy', model.val_accumulated_accuracy)

        validation_loss_weights = np.ones((config.num_classes, config.num_classes))
        mprint('Start Training ***********')
        best_acc = 0
        best_model_step = 0
        for current_iter in range(start_iter, config.max_iter):
            start_time_train = time.time()

            for mini_batch in range(config.caffe_iter_size - 1):
                feed_dict = {handle: training_handle}
                sess.run(accum_ops, feed_dict)

            feed_dict = {handle: training_handle}
            model_loss_value, accuracy_value, _ = sess.run([model.train_loss, model.train_accuracy, train_op],
                                                           feed_dict)

            if config.caffe_iter_size > 1:  ## Accumulated Gradient
                sess.run(zero_ops)

            train_time = time.time() - start_time_train


            if (current_iter % config.logging_threshold == 0):
                mprint(
                    'i {0:04d} loss {1:4f} Acc {2:2f} Batch Time {3:3f}'.format(current_iter, model_loss_value, accuracy_value,
                                                                                train_time))

                if (current_iter != 0):
                    run_metadata = tf.RunMetadata()
                    tf.local_variables_initializer().run()
                    sess.run(validation_iterator.initializer)

                    while True:
                        try:
                            feed_dict = {model.loss_class_weight: validation_loss_weights, handle: validation_handle}
                            val_loss_op, batch_accuracy, accuracy_op, _val_acc_op, _val_acc, c_cnf_mat = sess.run(
                                [val_loss, model.val_accuracy, model_acc_op, val_acc_op, model.val_accumulated_accuracy,
                                 model.val_confusion_mat], feed_dict)
                        except tf.errors.OutOfRangeError:
                            mprint('Val Acc {0} {1}'.format(_val_acc, batch_accuracy))
                            break



                    train_writer.add_run_metadata(run_metadata, 'step%03d' % current_iter)
                    train_writer.add_summary(val_loss_op, current_iter)
                    train_writer.add_summary(_val_acc_op, current_iter)
                    train_writer.add_summary(accuracy_op, current_iter)

                    train_writer.flush()


                    if (current_iter % config.logging_threshold == 0):
                        saver.save(sess, ckpt_file)
                        if best_acc < _val_acc:
                            saver.save(sess, ckpt_file + 'best')
                            best_acc = _val_acc
                            best_model_step = current_iter
                        ## Early dropping style.
                        mprint('Best Acc {0} at {1} == {2}'.format(best_acc, best_model_step, config.model_filename))

        saver.save(sess, ckpt_file)  ## Save final ckpt before closing
        ckpt = os.path.join(save_model_dir, str(current_iter), config.model_save_name)
        saver.save(sess, ckpt)
        sess.close()


if __name__ == '__main__':
    main()
