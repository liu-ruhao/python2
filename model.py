    # =========================================================================
import tensorflow as tf
    
class model(object):
    def __init__(self,sess,config,images_batch,labels):
        self.sess=sess
        self.train_writer = tf.summary.FileWriter(config["logs_train_dir"], sess.graph)
        
        self.batch_size=config["BATCH_SIZE"]
        self.n_classes=config["N_CLASSES"]
        self.learning_rate=config["learning_rate"]
        #self.saver = tf.train.Saver()
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        self.build(images_batch,labels)
# 训练操作定义  
    def build(self,images_batch,labels):
        #if ....
        #    self.structure_1(images_batch)
       # elif:
        #    self.structure_2(images_batch)
        self.structure_1(images_batch)
        self.evaluation(labels)
        self.losses(labels)
        self.opt()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
    # =========================================================================
    # 网络结构定义
    # 输入参数：images，image batch、4D tensor、tf.float32、[batch_size, width, height, channels]
    # 返回参数：logits, float、 [batch_size, n_classes]
    def structure_1(self,images):
        with tf.variable_scope('conv1') as scope:
            weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=1.0, dtype=tf.float32),
                                  name='weights', dtype=tf.float32)
    
            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[64]),
                                 name='biases', dtype=tf.float32)
    
            conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
        with tf.variable_scope('pooling1_lrn') as scope:
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
            norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
        with tf.variable_scope('conv2') as scope:
            weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 16], stddev=0.1, dtype=tf.float32),
                                  name='weights', dtype=tf.float32)
    
            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[16]),
                                 name='biases', dtype=tf.float32)
    
            conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name='conv2')
    
        with tf.variable_scope('pooling2_lrn') as scope:
            norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling2')
    
        with tf.variable_scope('local3') as scope:
            reshape = tf.reshape(pool2, shape=[self.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = tf.Variable(tf.truncated_normal(shape=[dim, 128], stddev=0.005, dtype=tf.float32),
                                  name='weights', dtype=tf.float32)
    
            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                                 name='biases', dtype=tf.float32)
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        with tf.variable_scope('local4') as scope:
            weights = tf.Variable(tf.truncated_normal(shape=[128, 128], stddev=0.005, dtype=tf.float32),
                                  name='weights', dtype=tf.float32)
    
            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                                 name='biases', dtype=tf.float32)
    
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
        with tf.variable_scope('softmax_linear') as scope:
            weights = tf.Variable(tf.truncated_normal(shape=[128, self.n_classes], stddev=0.005, dtype=tf.float32),
                                  name='softmax_linear', dtype=tf.float32)
    
            biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[self.n_classes]),
                                 name='biases', dtype=tf.float32)
            self.softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

    # -----------------------------------------------------------------------------
    # loss计算
    # 传入参数：logits，网络计算输出值。labels，真实值，在这里是0或者1
    # 返回参数：loss，损失值
    def losses(self, labels):
        with tf.variable_scope('loss') as scope:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.softmax_linear, labels=labels,
                                                                           name='xentropy_per_example')
            self.loss = tf.reduce_mean(cross_entropy, name='loss')
            tf.summary.scalar(scope.name + '/loss', self.loss)

    
    # --------------------------------------------------------------------------
    def summery(self,summary_op,step):
        summary_str = self.sess.run(summary_op)
        self.train_writer.add_summary(summary_str, step)
        
    def opt(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)
    def train(self):
        _, tra_loss, tra_acc = self.sess.run([self.train_op, self.loss, self.accuracy])
        return tra_loss, tra_acc
    # -----------------------------------------------------------------------

    def evaluation(self, labels):
        with tf.variable_scope('accuracy') as scope:
            correct = tf.nn.in_top_k(self.softmax_linear, labels, 1)
            correct = tf.cast(correct, tf.float16)
            self.accuracy = tf.reduce_mean(correct)
            tf.summary.scalar(scope.name + '/accuracy',self.accuracy)
    

