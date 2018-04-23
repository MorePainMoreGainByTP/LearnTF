import tensorflow as tf

class Model(object):
    """
    一个检测通用（多种）图像篡改操作的模型
    """
    def __init__(self,config):
        #config:一些参数配置
        #initializer是变量初始化的方式: tf.constant_initializer：常量初始化函数 tf.random_normal_initializer：正态分布
        #tf.truncated_normal_initializer：截取的正态分布 tf.random_uniform_initializer：均匀分布 tf.zeros_initializer：全部是0
        #tf.ones_initializer：全是1  tf.uniform_unit_scaling_initializer：满足均匀分布，但不影响输出数量级的随机值
        self.global_step = tf.get_variable("global_step",initializer=0,dtype=tf.int32,trainable=False)
        self.batch_size = config.batch_size
        self.decay = config.decay
        self.decay_step = config.decay_step
        self.starter_learning_rate = config.starter_learning_rate
        #输入图片尺寸227x227，颜色通道为1，即灰度图
        self.image_holder = tf.placeholder(tf.float32,shape=[self.batch_size,227,227,1])
        self.label_holder = tf.placeholder(tf.int32,[self.batch_size])
        self.keep_prob = tf.placeholder(tf.float32)
        #12个特殊卷积核，即Prediction Error Filters，训练时，参数被约束
        self.kernelRes = tf.placeholder(tf.float32,[5,5,1,12])
        self.num_classes = config.num_classes


    def print_activations(self,tensor):
        """打印tensor的名字以及尺寸"""
        #print(tensor.op.name,"\t",tensor.get_shape().as_list())
        pass

    def variable_with_weight_loss(self,shape,stddev,wl):
        """
        随机初始化weights，并考虑其对loss的影响
        将权重值考虑进loss中，以便得到更好的权重值，wl:比例系数，控制loss的大小
        """
        var = tf.Variable(tf.truncated_normal(shape,dtype=tf.float32,stddev=stddev))    #权重
        if wl is not None:
            weight_loss = tf.multiply(tf.nn.l2_loss(var),wl,name="weight_loss")  #对weight使用L2正则化来计算loss
            tf.add_to_collection("losses",weight_loss)  #将每个权重对应的loss统一保存到collection中以便最后计算总的loss

        return var

    def inference(self):
        """搭建CNN网络模型"""
        parameters = []

        #特殊卷积层
        with tf.name_scope("convRes") as scope:
            conv = tf.nn.conv2d(self.image_holder,self.kernelRes,strides=[1,1,1,1],padding="VALID") #尺寸由227*227变为223*223
            biasRes = tf.Variable(tf.constant(0.0,shape=[12],dtype=tf.float32),trainable=True,name="biases")

            self.print_activations(conv)
            self.print_activations(biasRes)

            parameters += [biasRes]

        convRes = tf.nn.bias_add(conv,biasRes,name=scope)
        self.print_activations(convRes)
        self.activation_summary(convRes)

        #conv1:第一卷积层
        with tf.name_scope("conv1") as scope:
            #wl=0：此权重不计算其loss，卷积核尺寸7*7，输入channel为12,输出64个feature map
            kernel = self.variable_with_weight_loss(shape=[7,7,12,64],stddev=5e-2,wl=0.0)
            conv = tf.nn.conv2d(convRes,kernel,strides=[1,2,2,1],padding="SAME")
            biases = tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),trainable=True,name="biases")
            bias = tf.nn.bias_add(conv,biases)
            conv1 = tf.nn.relu(bias,name=scope)
            parameters += [kernel,biases]

            self.print_activations(conv1)
            self.activation_summary(conv1)

        pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="pool1")
        lrn1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9,beta=0.75,name="lrn1")
        self.print_activations(pool1)
        self.print_activations(lrn1)

        #conv2：第二卷积层
        with tf.name_scope("conv2") as scope:
            kernel = self.variable_with_weight_loss(shape=[5,5,64,48],stddev=5e-2,wl=0.0)
            conv = tf.nn.conv2d(lrn1,kernel,strides=[1,1,1,1],padding="SAME")
            biases = tf.Variable(tf.constant(0.1,shape=[48],dtype=tf.float32),trainable=True,name="biases")
            bias = tf.nn.bias_add(conv,biases)
            conv2 = tf.nn.relu(bias,name=scope)
            parameters += [kernel,biases]

            self.print_activations(conv2)
            self.activation_summary(conv2)

        pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
        lrn2 = tf.nn.lrn(pool2,4,bias=1.0,alpha=0.001/9,beta=0.75,name="lrn2")

        self.print_activations(pool2)
        self.print_activations(lrn2)

        #fc1:全连接层第一层
        with tf.name_scope("fc1")  as scope:
            reshape = tf.reshape(lrn2,[self.batch_size,-1])
            dim = reshape.get_shape()[1].value
            weights = self.variable_with_weight_loss(shape=[dim,4096],stddev=0.04,wl=0.004)
            biases = tf.Variable(tf.constant(0.1,shape=[4096]),name="biases")
            fc1 = tf.nn.relu(tf.matmul(reshape,weights)+biases,name=scope)
            parameters += [weights,biases]

            self.print_activations(fc1)
            self.activation_summary(fc1)

        drop1 = tf.nn.dropout(fc1,self.keep_prob,name="drop1")  #dropput:50%
        self.print_activations(drop1)

        #fc2:全连接层第二层
        with tf.name_scope("fc2") as scope:
            weights = self.variable_with_weight_loss(shape=[4096,4096],stddev=0.04,wl=0.004)
            biases = tf.Variable(tf.constant(0.1,shape=[4096]),name="biases")
            fc2 = tf.nn.relu(tf.matmul(drop1,weights)+biases,name=scope)
            parameters += [weights,biases]

            self.print_activations(fc2)
            self.activation_summary(fc2)

        drop2 = tf.nn.dropout(fc2,self.keep_prob,name="drop2")
        self.print_activations(drop2)

        # fc3:全连接层第三层
        with tf.name_scope("fc3") as scope:
            weights = self.variable_with_weight_loss(shape=[4096, self.num_classes], stddev=1 / 4096.0, wl=0.0)
            biases = tf.Variable(tf.constant(0.1,shape=[self.num_classes]), name="biases")

        logits = tf.add(tf.matmul(drop2,weights),biases)    #不使用softmax处理就可以获得最终分类结果（直接比较inference输出的各类数值大小）
        self.print_activations(logits)
        self.activation_summary(logits)

        return logits,parameters

    def loss(self,logits):
        labels = tf.cast(self.label_holder,tf.int64)    #将前者类型转换为后者类型
        #对这个函数而言，tensorflow神经网络中是没有softmax层，而是在这个函数中进行softmax函数的计算
        #这里的logits通常是最后的全连接层的输出结果，labels是具体哪一类的标签，这个函数是直接使用标签数据的，而不是采用one-hot编码形式。
        cross_entropy_sum = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name="cross_entropy_per_example")
        cross_entropy_mean = tf.reduce_mean(cross_entropy_sum,name="cross_entropy")

        tf.add_to_collection("losses",cross_entropy_mean)    #将输出层误差与两个全连接层weights的L2 loss相加
        loss_value = tf.add_n(tf.get_collection("losses"),name="total_loss")

        tf.summary.scalar("loss",loss_value)

        return loss_value

    def train_op(self,total_loss):
        """训练:最小化loss"""
        #使用指数型衰减策略来减小学习率，计算公式：decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        #staircase 若为 False 则是标准的指数型衰减，True 时则是阶梯式的衰减方法
        #global_step 用于逐步计算衰减指数，decay_steps 用于决定衰减周期，decay_rate 是每次衰减的倍率
        learning_rate = tf.train.exponential_decay(self.starter_learning_rate,self.global_step,self.decay_step,self.decay,staircase=True)
        train_op = (tf.train.AdamOptimizer(learning_rate).minimize(total_loss,global_step = self.global_step))

        return train_op

    def cal_accuracy(self,logits):
        #logits：输出层直接输出，没有进行处理
        #返回的是batch_size的一维bool数组，代表了每一个样本是否预测准确即输出与label是相同的
        #而判断是否预测准确的策略就是in_top_k函数：若logits中每个样本得分最高的K的列/类，包含了target，则认为预测正确
        return tf.nn.in_top_k(logits,self.label_holder,1)

    def activation_summary(self,activation):
        name = activation.op.name
        tf.summary.histogram(name+"/activations",activation)    #直方图
        tf.summary.scalar(name+"/sparsity",tf.nn.zero_fraction(activation))

    def logits_summary(self,logits):
        pass

