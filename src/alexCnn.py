#coding:utf-8
import sys
import os
reload(sys)
sys.setdefaultencoding( "utf-8" )
import helper
import tensorflow as tf
import numpy as np


image_size=64
depth_1st_layer=32
depth_2rd_layer=64
batch_size = 128
fc1=1024
if(True):
    (XoB,YoB)=helper.storeAllPicInMe(flagOfO=False,flagOfB=True,size=64)
    (Xo0,Yo0)=helper.storeAllPicInMe(flagOfB=False,flagOfO=True,size=64)
    Xo0,Yo0=helper.shuffle(Xo0,Yo0)
    X=Xo0+XoB
    Y=Yo0+YoB
    dictory, Y = helper.convertDiscretToContinueT(Y)
    Y = helper.convertIndexTovector(len(dictory), Y)


    print "loading finished"
if(False):
    (X,Y)=helper.storeAllPicInMe(flagOfO=True,flagOfB=True)
    dictory, Y = helper.convertDiscretToContinueT(Y)
    Y = helper.convertIndexTovector(len(dictory), Y)
    X,Y=helper.shuffle(X,Y)





train_data_x=X[994:]
train_data_y=Y[994:]
test_data_x=X[0:994]
test_data_y=Y[0:994]

train_data_x,train_data_y=helper.shuffle(train_data_x,train_data_y)

num_batch=len(train_data_x)/batch_size

numOfCharacter=len(dictory)





X = tf.placeholder(tf.float32, [None,image_size,image_size ])
Y = tf.placeholder(tf.float32, [None, numOfCharacter])
##the parameter of dropout
keep_prob = tf.placeholder(tf.float32)


def chinese_hand_write_cnn():
    x = tf.reshape(X, shape=[-1, image_size, image_size, 1])
    # 3 conv layers
    w_c1 = tf.Variable(tf.random_normal([3, 3, 1, depth_1st_layer], stddev=0.01))
    b_c1 = tf.Variable(tf.zeros([depth_1st_layer]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    w_c2 = tf.Variable(tf.random_normal([3, 3, depth_1st_layer, depth_2rd_layer], stddev=0.01))
    b_c2 = tf.Variable(tf.zeros([depth_2rd_layer]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    """
    # 训练开始之后我就去睡觉了, 早晨起来一看, 白跑了, 准确率不足10%; 把网络变量改少了再来一发
    w_c3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    b_c3 = tf.Variable(tf.zeros([128]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    """

    # fully connect layer
    w_d = tf.Variable(tf.random_normal([(image_size/2/2)*(image_size/2/2)* depth_2rd_layer, fc1], stddev=0.01))
    b_d = tf.Variable(tf.zeros([fc1]))
    dense = tf.reshape(conv2, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(tf.random_normal([fc1, numOfCharacter], stddev=0.01))
    b_out = tf.Variable(tf.zeros([numOfCharacter]))
    out = tf.add(tf.matmul(dense, w_out), b_out)

    return out


def train_hand_write_cnn():
    output = chinese_hand_write_cnn()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1)), tf.float32))

    # TensorBoard
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 命令行执行 tensorboard --logdir=./log  打开浏览器访问http://0.0.0.0:6006
        summary_writer = tf.summary.FileWriter('./log', graph=tf.get_default_graph())

        for e in range(50):
            for i in range(num_batch):
                batch_x = train_data_x[i * batch_size: (i + 1) * batch_size]
                batch_y = train_data_y[i * batch_size: (i + 1) * batch_size]

                _, loss_, summary = sess.run([optimizer, loss, merged_summary_op],
                                             feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
                # 每次迭代都保存日志
                summary_writer.add_summary(summary, e * num_batch + i)
                print(e * num_batch + i, loss_)

                if (e * num_batch + i) % 100 == 0:
                    # 计算准确率
                    acc = accuracy.eval({X: test_data_x, Y: test_data_y, keep_prob: 1.})
                    # acc = sess.run(accuracy, feed_dict={X: text_data_x[:500], Y: text_data_y[:500], keep_prob: 1.})
                    print(e * num_batch + i, acc)


train_hand_write_cnn()