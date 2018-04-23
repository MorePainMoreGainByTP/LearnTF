import os
import tensorflow as tf
from PIL import Image

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 800*2
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 200*2


def read_and_decode(filename):
    """
    :param filename:  xxx.tfrecords
    :return:每次返回一个图像及其标签
    """
    # string_input_producer生成一个先入先出的队列，不限定读取数量, 文件阅读器用它来读取数据。
    filename_queue = tf.train.string_input_producer([filename])
    # 这个reader是符号化的，只有在sess中run才会执行。
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        "label": tf.FixedLenFeature([], tf.int64),  # 0D, 标量 ,若是一维数组 则[2]: # 1D，长度为2
        "image": tf.FixedLenFeature([], tf.string)  # 0D, 标量
    })
    # 解析从 serialized_example 读取到的内容
    image = tf.decode_raw(features["image"], tf.uint8)  # 对于BytesList，要重新进行解码，把string类型的0维Tensor变成uint8类型的1维Tensor。
    image = tf.reshape(image, [256, 256, 1])  # 把图像复原：256x256,1通道（灰度图）
    image = tf.random_crop(image, [227, 227, 1])  # 随机裁剪原图为227x227的大小
    # Randomly flip the image horizontally.
    #distorted_image = tf.image.random_flip_left_right(distorted_image)
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5   # normalize
    label = tf.cast(features['label'], tf.int64)

    return image, label


def train_input(data_dir, batch_size):
    """
    :param data_dir:存放训练数据的 .tfrecord 路径
    :param batch_size: 训练batch的大小
    :return:一个tensor
    """
    image, label = read_and_decode(data_dir)

    min_fraction_of_examples_in_queue = 0.9
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    num_preprocess_threads = 6  # 用于数据预处理的线程数

    # 使用shuffle_batch 随机打乱输入
    # 它是一种图运算，要跑在sess.run()里
    # 将tensor:[image, label]放进队列中，先打乱然后再从队列中取出batch_size个数的tensor组成一个batch，
    # 出列后需保持队列中剩余tensor个数大于min_after_dequeue
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],  # [single_image, single_label]
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples
    )

    return image_batch, label_batch


def test_input(data_dir, batch_size):
    image, label = read_and_decode(data_dir)
    min_fraction_of_examples_in_queue = 0.9
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    num_preprocess_threads = 6
    image_batch, label_batch = tf.train.batch(  # 同上，只不过不用打乱tensor的顺序
        [image, label],  # tensor:[image, label]就是代表单个的图像与标签数据，看成整体然后放入队列中
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

    return image_batch, label_batch
