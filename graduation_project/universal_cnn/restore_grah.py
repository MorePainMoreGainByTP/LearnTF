import cv2
import tensorflow as tf
from main import Config


def restore_kernel(config):
    with tf.Session() as sess:
        kernelRes = sess.run(tf.truncated_normal([5, 5, 1, 12], dtype=tf.float32, stddev=1e-1))
        # print("kernelRes:",kernelRes)
        with open(config.kernel_filename[1], "r") as f:
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


# 模型路径
model_path = 'grah_param/mnist.pb'
img_dir = r"src_data/"
class_index = r"1"

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    with open(model_path, "rb") as f:
        # 恢复模型
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            # 获得名字为x_input、output的tensor
            image_input = sess.graph.get_tensor_by_name("image_input:0")
            print("image_input shape:", image_input.get_shape())

            keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
            kernel_res = sess.graph.get_tensor_by_name("kernel_res:0")
            output = sess.graph.get_tensor_by_name("output:0")

            config = Config()
            kernel = restore_kernel(config)
            # img_value = np.array([img_value])
            img_list = []
            for i in range(16):
                img_path = img_dir + class_index + "/" + str(i + 1) + ".jpg"
                img_value = img_prepare(img_path)
                img_list.append(img_value)

            result = sess.run(output, feed_dict={
                image_input: img_list,
                keep_prob: 1.0,
                kernel_res: kernel})  # 利用训练好的模型预测结果
            print('模型预测结果为：\n', result)
