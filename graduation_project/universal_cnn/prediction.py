from main import Config
import tensorflow as tf
import model
import os
from PIL import Image, ImageFilter
import numpy as np
import cv2


def restore_kernel(config,class_index):
    with tf.Session() as sess:
        kernelRes = sess.run(tf.truncated_normal([5, 5, 1, 12], dtype=tf.float32, stddev=1e-1))
        # print("kernelRes:",kernelRes)
        with open(config.kernel_filename[class_index], "r") as f:
            for k in range(12):
                for i in range(5):
                    for j in range(5):
                        kernelRes[i, j, 0, k] = float(f.readline())
    return kernelRes


def img_prepare(img_path):
    sess = tf.Session()
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # print("original img:",img)
    image = tf.reshape(img, [256, 256, 1])  # 把图像复原：256x256,1通道（灰度图）
    # print("after reshape img:",image.eval(session=sess))
    image = tf.random_crop(image, [227, 227, 1])  # 随机裁剪原图为227x227的大小
    # print("after random_crop img:", image.eval(session=sess))
    # Randomly flip the image horizontally.
    # distorted_image = tf.image.random_flip_left_right(distorted_image)
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5  # normalize
    image = image.eval(session=sess)
    # print("after cast img:", image)
    return image


def predict(img_path, class_index, config):
    modeler = model.Model(config)
    # 构建卷积网络，返回输出层值logits以及每层的parameter
    logits, _ = modeler.inference()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        if os.path.exists("params/" + class_index + "/checkpoint"):
            saver.restore(sess, "params/" + class_index + "/model.ckpt")
            # print("parameters restore success!")
            kernel = restore_kernel(config,int(class_index))
            img_value = img_prepare(img_path)
            img_value = np.array([img_value])
            logits_value = sess.run(logits,
                                    feed_dict={modeler.image_holder: img_value,
                                               modeler.kernelRes: kernel,
                                               modeler.keep_prob: 1.0}
                                    )
            prediction = tf.argmax(logits, 1)  # 返回y_conv最大值的索引
            value = prediction.eval(feed_dict={
                modeler.image_holder: img_value,
                modeler.kernelRes: kernel,
                modeler.keep_prob: 1.0
            }, session=sess)
            print("value:",value)
            print("value[0]:",value[0])
            # print("value:", value, "\n", "logits_value:", logits_value)
            # record.append((value, logits_value))
            # display_list(record)
            return value[0]
        else:
            print("parameters restore fail!")
            return -1


def display_list(m_list):
    for i in range(len(m_list)):
        print("value:", m_list[i][0], "  logits:", m_list[i][1])


def main(img_dir, img_name, classes):
    config = Config()
    config.batch_size = 1
    config.num_classes = 2

    result = {}
    for index in classes:
        result[index] = predict(img_dir + img_name, index, config)
    print(result)
    return result


if __name__ == "__main__":
    main("src_data/1/", "1.jpg", ["1"])
