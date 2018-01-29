# encoding: utf-8

"""
@author: huguyuehuhu
@time: 17-9-23 下午10:53
"""
import tensorflow as tf
import os
import shutil


class MultiPerceptron:
    def __init__(self, train_param):
        # Network Parameters
        self.n_classes = 10  # MNIST total classes (0-9 digits)
        self.n_hidden_1 = 500  # 1st layer number of neurons
        self.n_hidden_2 = 256  # 2nd layer number of neurons
        self.n_input = 784  # MNIST data input (img shape: 28*28)
        # train param
        self.dataset_name = train_param['data_set']
        self.model_type = train_param['model_type']
        self.should_save_logs = train_param['should_save_logs']
        self.should_save_model = train_param['should_save_model']
        self.renew_logs = train_param['renew_logs']
        self.batches_step = 0

        # initial graph & session
        self._define_inputs()
        self._build_graph()
        self._initialize_session()

    ##################################################################
    # tf Graph input
    def _define_inputs(self):
        self.images = tf.placeholder(tf.float32, [None, self.n_input])
        self.labels = tf.placeholder(tf.float32, [None, self.n_classes])
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        # for batch_normalization
        self.is_training = tf.placeholder(tf.bool, shape=[])

    # tf Graph structure
    def _build_graph(self):
        n_input = self.n_input
        n_hidden_1 = self.n_hidden_1
        n_hidden_2 = self.n_hidden_2
        n_classes = self.n_classes
        images = self.images
        labels = self.labels
        learning_rate = self.learning_rate

        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            #'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
            'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(images, weights['h1']), biases['b1'])
        output_1 = tf.nn.tanh(layer_1)
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # Output fully connected layer with a neuron for each class
        #output = tf.matmul(layer_2, weights['out']) + biases['out']
        output = tf.matmul(output_1, weights['out']) + biases['out']
        # training accuracy
        prediction = tf.nn.softmax(output)
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # Define loss and optimizer
        self.loss_single_sample = tf.nn.softmax_cross_entropy_with_logits(
            logits=output, labels=labels)
        self.loss = tf.reduce_mean(self.loss_single_sample)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

        # self.updater = optimizer.minimize(self.loss)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.updater = optimizer.apply_gradients(grads_and_vars)

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        # restrict models GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf_version = float('.'.join(tf.__version__.split('.')[:2]))
        if tf_version <= 0.10:
            self.sess.run(tf.initialize_all_variables())
            logs_writer = tf.train.SummaryWriter
        else:
            self.sess.run(tf.global_variables_initializer())
            logs_writer = tf.summary.FileWriter
        self.saver = tf.train.Saver()
        self.summary_writer = logs_writer(self.logs_path)

    @property
    def count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        return total_parameters

    #################################################
    # log & save
    # model_identifier is models name, the only things to identify models log
    @property
    def model_identifier(self):
        return "{}_dataset_{}".format(
            self.model_type, self.dataset_name)

    # path are ./saves/
    @property
    def save_path(self):
        try:
            save_path = self._save_path
        except AttributeError:
            save_path = 'saves/%s' % self.model_identifier
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'models.chkpt')
            self._save_path = save_path
        return save_path

    # path are ./logs/
    @property
    def logs_path(self):
        try:
            logs_path = self._logs_path
        except AttributeError:
            logs_path = 'logs/%s' % self.model_identifier
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return logs_path

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path + 'something')
        except Exception as e:
            raise IOError("Failed to to load models "
                          "from save path: %s" % self.save_path)
        self.saver.restore(self.sess, self.save_path)
        print("Successfully load models from save path: %s" % self.save_path)

    # log (loss & accuracy) with tag and postfix(prefix: per_epoch;train;valid)
    def log_loss_accuracy(self, loss, accuracy, epoch, prefix,
                          should_print=True):
        if should_print:
            print("mean cross_entropy: %f, mean accuracy: %f" % (
                loss, accuracy))
        # this way is more flexible faster than tf.summary.scalar;
        # http: // tang.su / 2017 / 01 / manually - create - summary /
        # https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/core/framework/summary.proto
        # https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514#file-tensorboard_logging-py-L41
        # also for (imge,hist)

        summary = tf.Summary(value=[
            tf.Summary.Value(
                tag='loss_%s' % prefix, simple_value=float(loss)),
            tf.Summary.Value(
                tag='accuracy_%s' % prefix, simple_value=float(accuracy))
        ])
        self.summary_writer.add_summary(summary, epoch)