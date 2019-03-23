import tensorflow as tf
import math
import constants as const

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_SCALE_FACTOR = 0.017

def _std_image_normalize(image, stds):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  num_channels = image.get_shape().as_list()[-1]
  if len(stds) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=3, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] /= stds[i]
  return tf.concat(axis=3, values=channels)

def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=3, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=3, values=channels)


def denseNet_preprocess(images):
    images = tf.image.convert_image_dtype(images, dtype=tf.float32) ## from uint8[0 255] ==> float [0 1]

    ## Adapted from pytorch normalization.
    # I assumed it equivalent to the following [0 255] range normalization
    images = _mean_image_subtraction(images, [0.485, 0.456, 0.406])
    images = _std_image_normalize(images, [0.229, 0.224, 0.225])

    #images= _mean_image_subtraction(images, [_R_MEAN, _G_MEAN, _B_MEAN])
    #images = images * _SCALE_FACTOR;
    return images

def inception_preprocessing(images):
    #print(images)
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)
    images = tf.subtract(images, 0.5)
    images = tf.multiply(images, 2.0)

    return images

def center_crop(images,preprocess_func ): ## Used during evaluation
    center_offest = (256 - const.frame_width )//2 # I already resized all images to 256
    images = tf.image.crop_to_bounding_box(images, center_offest , center_offest , const.frame_height, const.frame_width)

    if preprocess_func == 'inception_v1':
        print('Inception Format Augmentation')
        images = inception_preprocessing(images)
    elif preprocess_func == 'densenet':
        print('DenseNet Format Augmentation')
        images = denseNet_preprocess(images)
    else:
        raise NotImplementedError()

    return images

def augment(images,
            preprocess_func,
            resize=None,  # (width, height) tuple or None
            horizontal_flip=False,
            vertical_flip=False,
            rotate=0,  # Maximum rotation angle in degrees
            noise_probability = 0,
            color_aug_probability = 0,
            crop_probability=0,  # How often we do crops
            crop_min_percent=0.6,  # Minimum linear dimension of a crop
            crop_max_percent=1.,  # Maximum linear dimension of a crop
            mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf  ## Used during training

    ## Credit goes to  https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19

    ## Always assume image [0,255]

    # Random Crop on Batch Level
    max_offest = 256 - const.frame_width # I already resized all images to 256
    rand = tf.random_uniform([2], minval=0, maxval=max_offest,dtype=tf.int32)
    height_offset = tf.cast(rand[0] , dtype=tf.int32)
    width_offest = tf.cast(rand[1] , dtype=tf.int32)
    images = tf.image.crop_to_bounding_box(images,height_offset, width_offest, const.frame_height , const.frame_width )


    if preprocess_func == 'densenet':
        print('DenseNet Format Augmentation')
        images = denseNet_preprocess(images)
    elif preprocess_func == 'inception_v1':
        print('Inception Format Augmentation')
        images = inception_preprocessing(images)
    else:
        raise NotImplementedError()



    with tf.name_scope('augmentation'):
        shp = tf.shape(images)
        batch_size, height, width = shp[0], shp[1], shp[2]
        width = tf.cast(width, tf.float32)
        height = tf.cast(height, tf.float32)

        # The list of affine transformations that our image will go under.
        # Every element is Nx8 tensor, where N is a batch size.
        transforms = []
        identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        if horizontal_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if vertical_flip:
            coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
            flip_transform = tf.convert_to_tensor(
                [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
            transforms.append(
                tf.where(coin,
                         tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if rotate > 0:
            angle_rad = rotate / 180 * math.pi
            angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
            transforms.append(
                tf.contrib.image.angles_to_projective_transforms(
                    angles, height, width))

        if crop_probability > 0:
            crop_pct = tf.random_uniform([batch_size], crop_min_percent,
                                         crop_max_percent)
            left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
            top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
            crop_transform = tf.stack([
                crop_pct,
                tf.zeros([batch_size]), top,
                tf.zeros([batch_size]), crop_pct, left,
                tf.zeros([batch_size]),
                tf.zeros([batch_size])
            ], 1)

            coin = tf.less(
                tf.random_uniform([batch_size], 0, 1.0), crop_probability)
            transforms.append(
                tf.where(coin, crop_transform,
                         tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

        if transforms:
            images = tf.contrib.image.transform(
                images,
                tf.contrib.image.compose_transforms(*transforms),
                interpolation='BILINEAR')  # or 'NEAREST'

        def cshift(values):  # Circular shift in batch dimension
            return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

    #resize = (const.frame_height, const.frame_width)
    #images = tf.image.resize_bilinear(images, resize)
    return images

