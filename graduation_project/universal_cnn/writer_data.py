import os
import tensorflow as tf
from PIL import Image


def create_record(srcDir):
    """
    创建一个专门存储tensorflow数据的writer，扩展名为’.tfrecord’。该文件中依次存储着序列化的tf.train.Example类型的样本。
    将训练与测试图像原始数据与label写入格式化文件xxxx.tfrecords中方便使用
    :param srcDir:存放原始图像(.jpg)的路径
    """
    train_writer = tf.python_io.TFRecordWriter("./train_img.tfrecords")
    test_writer = tf.python_io.TFRecordWriter("./test_img.tfrecords")

    classes = os.listdir(srcDir)  # srcDir路径下放的是分类文件夹，各文件夹(命名为：0,1,2,3,...)下是图片img_xx.jpg
    print("类文件夹： ",classes)

    for index, name in enumerate(classes):  #下标与文件名 classes：list对象
        class_path = srcDir + name + "/"  # 分类文件夹路径
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            image = img.tobytes()  # 图像的二进制数据
            # 每一张img及其label对应一个example
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
            }))
            # feature dict接受三种类型数据： tf.train.FloatList：列表每个元素为float
            # tf.train.Int64List：列表每个元素为int64
            # tf.train.BytesList：列表每个元素为字节字符串   第三种类型尤其适合图像样本。注意在转成字符串之前要设定为uint8类型。

            # 根据img的名称判断是训练数据还是测试数据,同一篡改类型的image训练与测试数据放在一起，根据image的id来区分
            img_id = int(img_name[0:len(img_name) - 4])
            if img_id <= 800:   #前800张图片用于训练，后200张用于测试
                train_writer.write(example.SerializeToString())  # 序列化为字符串
            else:
                test_writer.write(example.SerializeToString())

            print(img_path, index, img_id)

    train_writer.close()
    test_writer.close()


'''
#以下代码时简单读取 tfrecord文件
for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    image = example.features.feature['image'].bytes_list.value
    label = example.features.feature['label'].int64_list.value
    # 可以做一些预处理之类的
    print image, label
'''
