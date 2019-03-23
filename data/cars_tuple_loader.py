from data.base_tuple_loader import BaseTupleLoader
import numpy as np

class CarsTupleLoader(BaseTupleLoader):

    def __init__(self,args):
        BaseTupleLoader.__init__(self,args)

        self.img_path = args['db_path'] + '/'

        lbls = self.data_df['label']
        lbl2idx = np.sort(np.unique(lbls))

        self.lbl2idx_dict = {k: v for v, k in enumerate(lbl2idx)}
        self.final_lbls = [self.lbl2idx_dict[x] for x in list(lbls.values)]

        self.num_classes = len(self.lbl2idx_dict.keys())
        self.data_idx = 0


        print('Data size ', self.data_df.shape[0], 'Num lbls', len(self.lbl2idx_dict.keys()))


if __name__ == '__main__':

    args = dict()
    args['csv_file'] = config.train_csv_file
    train_iter = CarsTupleLoader(args)
    args['csv_file'] = config.test_csv_file
    val_iter = CarsTupleLoader(args)

    train_imgs, train_lbls = train_iter.imgs_and_lbls()
    val_imgs, val_lbls = val_iter.imgs_and_lbls()
    import tensorflow as tf
    from data.tf_tuple_loader import TensorflowTupleLoader
    train_dataset = TensorflowTupleLoader(train_imgs, train_lbls, is_training=True).dataset
    handle = tf.placeholder(tf.string, shape=[])


    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    images_ph, lbls_ph = iterator.get_next()

    training_iterator = train_dataset.make_one_shot_iterator()
    sess = tf.InteractiveSession()
    training_handle = sess.run(training_iterator.string_handle())

    imgs, lbls = sess.run([images_ph,lbls_ph],{handle:training_handle})
    print(imgs.shape,lbls.shape)
    print(np.min(imgs),np.max(imgs))
