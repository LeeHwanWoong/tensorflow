import os
import time
import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import re
import random


def cut(contents, cut=2):
    results = []
    for content in contents:
        words = content.split()
        result = []
        for word in words:
            result.append(word[:cut])
            #모든 단어의 앞으 두개만 잘라서 사용
        results.append(' '.join([token for token in result]))
    return results

####################################################
# divide train/test set function                   #
####################################################
#단순히 트레이닝용 테스트용 비율로 나누는것
def divide(x, y, train_prop):
    random.seed(1234)
    x = np.array(x)
    y = np.array(y)
    tmp = np.random.permutation(np.arange(len(x)))
    x_tr = x[tmp][:int(round(train_prop * len(x)))]
    y_tr = y[tmp][:int(round(train_prop * len(x)))]
    x_te = x[tmp][-int((len(x)-round(train_prop * len(x)))):]
    y_te = y[tmp][-int((len(x)-round(train_prop * len(x)))):]
    return x_tr, x_te, y_tr, y_te


####################################################
# making input function                            #
####################################################
#x값 전처리
def make_input(documents, max_document_length):

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(documents)))

    # Extract word:id mapping from the object.

    vocab_dict = vocab_processor.vocabulary_._mapping
    # Sort the vocabulary dictionary on the basis of values(id).
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    # Treat the id's as index into list and create a list of words in the ascending order of id's
    # word with id i goes at index i of the list.
    vocabulary = list(list(zip(*sorted_vocab))[0])
    return x, vocabulary, len(vocab_processor.vocabulary_)

####################################################
# make output function                             #
####################################################
def make_output(points, threshold):
    results = np.zeros((len(points),10))
    for idx, point in enumerate(points):
        results[idx,point] = 1
    return results

####################################################
# check maxlength function                         #
####################################################
def check_maxlength(contents):
    max_document_length = 0
    for document in contents:
        document_length = len(document.split())
        if document_length > max_document_length:
            max_document_length = document_length
    return max_document_length

####################################################
# loading function                                 #
####################################################
def loading_rdata(data_path = 'train2.csv', eng=True, num=True, punc=False):
    corpus = pd.read_csv(open(data_path,'rU'), sep=",", encoding="utf-8")
    corpus = np.array(corpus)
    contents = []
    points = []
    for idx,doc in enumerate(corpus):
        if isNumber(doc[0]) is False:
            contents.append(doc[0])
            points.append(doc[1])
        if idx % 100000 is 0:
            print('%d docs / %d save' % (idx, len(contents)))
    return contents, points

def isNumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

class TextCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size,embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        """
        <Variable>
        """
        #with tf.device('/gpu:1'), tf.name_scope("embedding"):
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            #build_my_model()
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
            #w가 전체 단어를 만들어 놓은 테이블
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            #200*128짜리 하나 즉 문장 하나
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            #차원만 변경

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        #filter_sizes = [3,4,5] -> conv, relu,pool이 각각 3개씩이고 두께는 3,4,5순
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                #filter_size * filter_size * num_filters
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                #ksize = kernel size
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        #num_filters = 128 /len(filter_sizes) = 3
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs,3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            #num_classes가 레이블 종류
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.scores)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

# data loading
#contents = text, points = rating

contents, points = loading_rdata()
#contents, point 모두 배열
contents = cut(contents,cut=2)
#결과는 모든 단어 2개씩만 남기고 2중배열 리턴

# tranform document to vector
max_document_length = 200
x, vocabulary, vocab_size = make_input(contents,max_document_length)
#배열로 나옴
'''
['<UNK>', 'This', 'is', 'a', 'cat', 'must', 'be', 'boy', 'dog']
[[1 2 3 4 0]
 [1 5 6 7 0]
 [1 2 3 3 8]]
 '''
print('words : %s' % (vocab_size))
y = make_output(points,threshold=2.5)

# divide dataset into train/test set
x_train, x_test, y_train, y_test = divide(x,y,train_prop=0.8)
filex = open('temp_x.txt','w')
x_train.tofile(filex)
filey = open('temp_y.txt','w')
y_train.tofile(filey)

# Model Hyperparameters

# 3. train the model and test
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        cnn = TextCNN(sequence_length=x_train.shape[1],
                      num_classes=y_train.shape[1],
                      vocab_size=vocab_size,
                      embedding_size=128,
                      filter_sizes=list(map(int, "3,4,5".split(","))),
                      num_filters=128,
                      l2_reg_lambda=0.1)
        #y_train.shape[1] = 10

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 0.5
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)


        def batch_iter(data, batch_size, num_epochs, shuffle=True):
            """
            Generates a batch iterator for a dataset.
            """
            data = np.array(data)
            data_size = len(data)
            num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
            for epoch in range(num_epochs):
                # Shuffle the data at each epoch
                if shuffle:
                    shuffle_indices = np.random.permutation(np.arange(data_size))
                    shuffled_data = data[shuffle_indices]
                else:
                    shuffled_data = data
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size)
                    yield shuffled_data[start_index:end_index]


        # Generate batches
        batches = batch_iter(
            list(zip(x_train, y_train)), 64, 10)

        testpoint = 0
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 100 == 0:
                if testpoint + 100 < len(x_test):
                    testpoint += 100
                else:
                    testpoint = 0
                print("\nEvaluation:")
                dev_step(x_test[testpoint:testpoint+100], y_test[testpoint:testpoint+100], writer=dev_summary_writer)
                print("")
            if current_step % 100 == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))