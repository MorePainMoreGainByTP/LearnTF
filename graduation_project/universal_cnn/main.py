import writer_data as writer
import reader_data as reader
import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import time
import math
import model


class Config(object):
    """配置类，设置各种参数值"""
    # 训练与测试数据存放路径
    data_path = "./src_data/"

    num_classes = 2
    batch_size = 32
    max_step = 6000  # 迭代训练次数

    decay = 0.95
    decay_step = 200
    starter_learning_rate = 1e-3

    # 变量保存与恢复
    # steps = max_step
    # checkpoint_iter = 1000
    param_dir = "./params/"
    save_filename = "model.ckpt"
    # load_filename = "model-" + str(steps)

    # 总结
    log_dir = "./log/"
    summary_iter = 1000
    kernel_filename = ['', "kernel/1.txt", "kernel/2.txt", "kernel/3.txt", "kernel/4.txt"]


def init_kernelRes(sess):
    '''
    创建第一卷积层的卷积核，并进行特殊处理
    '''
    kernelRes = sess.run(tf.truncated_normal([5, 5, 1, 12], dtype=tf.float32, stddev=1e-1))
    k_sum = sess.run(tf.reduce_sum(kernelRes, reduction_indices=[0, 1, 2]))
    # reduction_indices:指定哪些维求和，0，1,2分别指：第一、二、三维，则对应5x5x1，所以是把12个weight的权分别求和，得到12个元素的一维数组

    for k in range(12):
        for i in range(5):
            for j in range(5):
                if i != 2 or j != 2:
                    kernelRes[i, j, 0, k] /= (k_sum[k] - kernelRes[2, 2, 0, k])  # 将除中心外的权重，求和再相除，使得这些权重之和为1
        kernelRes[2, 2, 0, k] = -1  # 将中心权重设为 -1

    return kernelRes


def main():
    config = Config()

    # 对数据进行预处理
    writer.create_record(config.data_path)

    image_train, label_train = reader.train_input(data_dir="./train_img.tfrecords", batch_size=config.batch_size)
    image_test, label_test = reader.test_input(data_dir="./test_img.tfrecords", batch_size=config.batch_size)

    modeler = model.Model(config)
    # 构建卷积网络，返回输出层值logits以及每层的parameter
    logits, _ = modeler.inference()
    # 定义loss
    loss = modeler.loss(logits)
    # 选择优化器，学习率减小策略
    train_op = modeler.train_op(loss)
    # 选择预测正确判定策略
    top_k = modeler.cal_accuracy(logits)

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()  # max_to_keep:最多保存多少份ckpt，新的覆盖旧的
    #max_to_keep=math.ceil(config.max_step / config.checkpoint_iter)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True  # 若指定的设备不存在,tensorFlow 会自动选择一个存在并且支持的设备来运行 operation.

    with tf.Session(config=sess_config) as sess:
        sess.run(init)
        # Coordinator(协调者)类可以用来同时停止多个工作线程并且向那个在等待所有工作线程终止的程序报告异常
        # Coordinator类用来帮助多个线程协同工作，多个线程同步终止
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        kernelRes_train = init_kernelRes(sess)

        merged = tf.summary.merge_all()
        # tensorboard-可视化CNN
        logdir = os.path.join(config.log_dir, datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
        train_writer = tf.summary.FileWriter(logdir, sess.graph)

        # 训练
        print("\n----------------开始训练----------------\n")
        for step in range(config.max_step):
            k_sum = sess.run(tf.reduce_sum(kernelRes_train, reduction_indices=[0, 1, 2]))
            # reduction_indices:指定压缩哪些维，0，1,2分别指：第一、二、三维，则对应5x5x1，所以是把12个weight的权分别求和，得到12个元素的一维数组
            for k in range(12):
                for i in range(5):
                    for j in range(5):
                        if i != 2 or j != 2:
                            kernelRes_train[i, j, 0, k] /= (
                                    k_sum[k] - kernelRes_train[2, 2, 0, k])  # 将除中心外的权重，求和再相除，使得这些权重之和为1
                kernelRes_train[2, 2, 0, k] = -1  # 将中心权重设为 -1

            start_time = time.time()
            # with tf.device 创建一个设备环境, 这个环境下的 operation 都统一运行在环境指定的设备上.
            # 若一个operation在CPU与GPU上都可执行，则GPU优先，若存在多个GPU，则优先使用ID小的GPU
            with tf.device("/cpu:0"):
                image_batch, label_batch = sess.run([image_train, label_train])
                feed_dict = {
                    modeler.image_holder: image_batch,
                    modeler.label_holder: label_batch,
                    modeler.kernelRes: kernelRes_train,
                    modeler.keep_prob: 0.5
                }

            with tf.device("/gpu:0"):
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            duration = time.time() - start_time
            # 输出每次迭代耗时
            if step % 100 == 0:
                examples_per_sec = config.batch_size / duration
                sec_per_batch = float(duration)
                format_str = ("step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)")
                print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
                # print("logits_value:",logits_value)

            with tf.device("/cpu:0"):
                # write summary
                if (step + 1) % config.summary_iter == 0:
                    summary = sess.run(merged, feed_dict=feed_dict)
                    train_writer.add_summary(summary, modeler.global_step.eval())
        print("\n----------------结束训练----------------\n")
        with tf.device("/cpu:0"):
            # 保存parameter即checkpoint
            saver.save(sess, config.param_dir + config.save_filename)
            # try:
            #     print(kernelRes_train)
            #     print(kernelRes_train.eval(session=sess))
            #     print(kernelRes_train.get_shape())
            # except:
            #     pass

        print("\n----------------开始测试----------------\n")
        # 测试
        num_examples = reader.NUM_EXAMPLES_PER_EPOCH_FOR_TEST
        num_iter = int(math.ceil(num_examples / config.batch_size))
        true_count = 0
        total_sample_count = num_iter * config.batch_size
        accuracy = np.zeros(config.num_classes)
        step = 0
        print("num_iter:%d,total_sample_count:%d" % (num_iter, total_sample_count))
        while step < num_iter:
            print("step:",step)
            #sssss
            with tf.device("/cpu:0"):
                image_batch, label_batch = sess.run([image_test, label_test])
                # print("image_batch:",image_batch)
                # print("label_batch:",label_batch)
            with tf.device("/gpu:0"):
                predictions ,logits_value= sess.run([top_k,logits],
                                       feed_dict={modeler.image_holder: image_batch,
                                                  modeler.label_holder: label_batch,
                                                  modeler.kernelRes: kernelRes_train,
                                                  modeler.keep_prob: 1.0})
                print("predictions:",predictions)
                print("logits_value:",logits_value)
            true_count += np.sum(predictions)
            # print("predictions:", predictions)
            for i in range(config.batch_size):
                if predictions[i]:
                    # print("%s +1" % label_batch[i])
                    accuracy[label_batch[i]] += 1
            print("current accuracy:", accuracy)
            # print("true_count:%s,\ttotal_sample_count:%s" % (true_count, total_sample_count))
            step += 1
        print("\n----------------结束测试----------------\n")

        print("=================================================")
        print("true_count:%d,total_sample_count:%d" % (true_count, total_sample_count))
        precision = 1.0 * true_count / total_sample_count
        print("总体正确率 precision @ l = %.3f" % precision)

        # accuracy = accuracy * config.num_classes / total_sample_count
        # accuracy = map(lambda x: x / 200, accuracy)
        first = accuracy[0] / (200 + total_sample_count - num_examples)
        accuracy = accuracy / 200
        accuracy[0] = first
        print("分类准确率:", accuracy)
        print("=================================================")

        with open(config.kernel_filename[1], "w") as f:
            for k in range(12):
                for i in range(5):
                    for j in range(5):
                        f.write(str(kernelRes_train[i, j, 0, k]) + "\n")
        # print("kernelRes_train：", kernelRes_train)
        coord.request_stop()  # request_stop(): 请求该线程停止。 其他线程的should_stop()将会返回True，然后都停下来。
        coord.join(threads)  # join(<list of threads>):等待被指定的线程终止。


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("迭代%d次，总耗时%d秒" % (Config().max_step, end_time - start_time))
