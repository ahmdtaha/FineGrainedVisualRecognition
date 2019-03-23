from pydoc import locate
import tensorflow as tf
import constants as const
from nets import nn_utils

class TensorflowTupleLoader:

    def __init__(self,imgs,lbls ,cfg,is_training,repeat=True,batch_size=None):
        self.dataset = self.dataset_from_files(imgs, lbls,cfg,is_training,repeat=repeat,batch_size=batch_size)


    def dataset_from_files(self,train_imgs, train_lbls,cfg,is_training,repeat=True,batch_size=None):

        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string,channels=3) ## uint8 image
            return image_decoded, tf.one_hot(label, cfg.num_classes,dtype=tf.int64)

        filenames = tf.constant(train_imgs)
        labels = tf.constant(train_lbls,dtype=tf.int32)


        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

        if repeat:
            dataset = dataset.shuffle(len(train_imgs))
            dataset = dataset.repeat(None)
        else:
            #dataset = dataset.shuffle(len(train_imgs)) ## Avoid shuffle to test batch normalization
            dataset = dataset.repeat(1)


        dataset = dataset.map(_parse_function,num_parallel_calls=3)

        ## Image level augmentation. It is possible to use it but I find batch level augmentation better
        # preprocess_mod = locate(config.preprocessing_module)
        # func_name = preprocess_mod.preprocess_for_train_simple
        # if not is_training:
        #     func_name = preprocess_mod.preprocess_for_eval_simple
        # dataset = dataset.map(lambda im, lbl,weight: (func_name (im,const.frame_height,const.frame_width), lbl,weight))

        if batch_size is None:
            batch_size = cfg.batch_size
        else:
            print('Eval Batch used')


        dataset = dataset.batch(batch_size)

        ## Batch Level Augmentation
        if is_training:
            dataset = dataset.map(lambda im_batch, lbl_batch: (nn_utils.augment(im_batch,cfg.preprocess_func,
                                                           resize=(const.frame_height, const.frame_width),horizontal_flip=True, vertical_flip=False, rotate=0, crop_probability=0,crop_min_percent=0), lbl_batch))
        else:
            dataset = dataset.map(lambda im_batch, lbl_batch: (nn_utils.center_crop(im_batch,cfg.preprocess_func),lbl_batch))


        dataset = dataset.prefetch(1)
        return dataset



