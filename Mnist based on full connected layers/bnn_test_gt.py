# -*- coding: utf-8 -*-
"""
##################################################################
     Bayesian Neural Network based on 4 full-connected layers

     time:2019.3.29
     author: Yingh Liu

data sets: MNIST
just for test
#################################################################
"""
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Categorical, Normal, OneHotCategorical
import edward as ed
import pandas as pd
import struct
mnist =input_data.read_data_sets(r"~/MNIST_data/",one_hot=True)
file_w_b =r"~/weights.bat"
#mnist =input_data.read_data_sets("MNIST_data/",one_hot=True)
print (" number of trian data is %d" % (mnist.train.num_examples))
print (" number of test data is %d" % (mnist.test.num_examples))

ed.set_seed(314159)

N = 100 #minibatch nunber of images
D = 784 #Mnist 1 frame 28*28=784
K = 10  # 10 classes ,figure 0~9
filers_0 = 400# layer1 filters number
"""
2 layers full connected neural network

Inputs are the logits, not probabilities.
Args:
  features  :input features
  W_0 W_1 :weights in layer 
  b_0 b_1 :bias in layer
Returns: 
  
"""
def two_layers_NN(feature, W_0, W_1,b_0, b_1):
    h = tf.nn.tanh(tf.matmul(feature, W_0) + b_0)
    #h = tf.matmul(feature, W_0) + b_0
    h = tf.matmul(h, W_1) + b_1
    return h
"""
def four_layers_NN(feature, W_0, W_1, W_2, W_3,
                   b_0, b_1, b_2, b_3):
    h = tf.nn.relu(tf.matmul(feature, W_0) + b_0)
    h = tf.nn.relu(tf.matmul(h, W_1) + b_1)
    h = tf.nn.relu(tf.matmul(h, W_2) + b_2)
    h = tf.matmul(h, W_3) + b_3
    return h
def three_layers_NN(feature, W_0, W_1,W_2,b_0, b_1,b_2):
    h = tf.nn.tanh(tf.matmul(feature, W_0) + b_0)
    #h = tf.nn.tanh(tf.matmul(h, W_1) + b_1)
    h = tf.matmul(h, W_1) + b_1
    h = tf.matmul(h, W_2) + b_2
    return h
"""
#Save weights and bias
def save_w_b(W_0,W_1,b_0,b_1):
    with open(file_w_b, 'wb') as fw:
        fw.seek(0)  # 移动文件指针
        fw.truncate()  # 清空文件内容
        np.set_printoptions(threshold=10000000)#防止输出省略号

        #fw.write((np.shape(W_0.mean().eval())))
        fw.write((W_0.mean().eval()).astype(np.float32))
        fw.write((W_0.variance().eval()).astype(np.float32))
        fw.write((W_1.mean().eval()).astype(np.float32))
        fw.write((W_1.variance().eval()).astype(np.float32))

        #fw.write(str(np.shape(b_0.mean().eval())))
        fw.write((b_0.mean().eval()).astype(np.float32))
        fw.write((b_0.variance().eval()).astype(np.float32))
        fw.write((b_1.mean().eval()).astype(np.float32))
        fw.write((b_1.variance().eval()).astype(np.float32))
        
#reload weights and bias
def load_w_b():
    with open(file_w_b, 'rb') as fw:
        data_raw = struct.unpack("f" * D * filers_0, fw.read(4 * D * filers_0))#float32类型,4个字节
        qw0_mean = tf.convert_to_tensor(np.asarray(data_raw).reshape(D, filers_0),dtype=np.float32)
        data_raw = struct.unpack("f" * D * filers_0, fw.read(4 * D * filers_0))  # float32类型,4个字节
        qw0_variance = tf.convert_to_tensor(np.asarray(data_raw).reshape(D, filers_0),dtype=np.float32)
        
        data_raw = struct.unpack("f" * filers_0 * K, fw.read(4 * filers_0 * K))#float32类型,4个字节
        qw1_mean = tf.convert_to_tensor(np.asarray(data_raw).reshape(filers_0, K),dtype=np.float32)
        data_raw = struct.unpack("f" * filers_0 * K, fw.read(4 * filers_0 * K))  # float32类型,4个字节
        qw1_variance = tf.convert_to_tensor(np.asarray(data_raw).reshape(filers_0, K),dtype=np.float32)

        data_raw = struct.unpack("f" * filers_0, fw.read(4 * filers_0))  # float32类型,4个字节
        qb0_mean = tf.convert_to_tensor(np.asarray(data_raw).reshape(filers_0),dtype=np.float32)
        data_raw = struct.unpack("f" * filers_0, fw.read(4 * filers_0))  # float32类型,4个字节
        qb0_variance = tf.convert_to_tensor(np.asarray(data_raw).reshape(filers_0),dtype=np.float32)
        data_raw = struct.unpack("f" * K, fw.read(4 * K))  # float32类型,4个字节
        qb1_mean = tf.convert_to_tensor(np.asarray(data_raw).reshape(K),dtype=np.float32)
        data_raw = struct.unpack("f" * K, fw.read(4 * K))  # float32类型,4个字节
        qb1_variance = tf.convert_to_tensor(np.asarray(data_raw).reshape(K),dtype=np.float32)

        qw0 = Normal(loc=tf.Variable(qw0_mean),
                    scale=tf.Variable(qw0_variance))
        qw1 = Normal(loc=tf.Variable(qw1_mean),
                    scale=tf.Variable(qw1_variance))
        qb0 = Normal(loc=tf.Variable(qb0_mean),
                    scale=tf.Variable(qb0_variance))
        qb1 = Normal(loc=tf.Variable(qb1_mean),
                    scale=tf.Variable(qb1_variance))
        tf.global_variables_initializer().run()

        return qw0,qw1,qb0,qb1

x =tf.placeholder(tf.float32,[None,D])
# Initialize the weights and bias
W_0 = Normal(loc=tf.zeros([D, filers_0]), scale=tf.ones([D, filers_0]))
W_1 = Normal(loc=tf.zeros([filers_0, K]), scale=tf.ones([filers_0, K]))
b_0 = Normal(loc=tf.zeros(filers_0), scale=tf.ones(filers_0))
b_1 = Normal(loc=tf.zeros(K), scale=tf.ones(K))

y = Categorical(logits=two_layers_NN(x, W_0, W_1, b_0, b_1),name="y")

#initialize the variantional inference  weights and bias
qW_0 = Normal(loc=tf.get_variable("qW_0/loc", [D, filers_0]),
              scale=tf.nn.softplus(tf.get_variable("qW_0/scale", [D, filers_0])))
qW_1 = Normal(loc=tf.get_variable("qW_1/loc", [filers_0, K]),
              scale=tf.nn.softplus(tf.get_variable("qW_1/scale", [filers_0, K])))
qb_0 = Normal(loc=tf.get_variable("qb_0/loc", [filers_0]),
              scale=tf.nn.softplus(tf.get_variable("qb_0/scale", [filers_0])))
qb_1 = Normal(loc=tf.get_variable("qb_1/loc", [K]),
              scale=tf.nn.softplus(tf.get_variable("qb_1/scale", [K])))
#Loss function
y_ph =tf.placeholder(tf.int32,[N])
#inference =ed.KLqp({w:qw,b:qb},data={y:y_ph})
inference =ed.KLqp({W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1},data={y:y_ph})

inference.initialize(n_iter=5000,n_print=100,#yong
                     scale={y:float(mnist.train.num_examples)/N})
sess =tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
    X_batch, Y_batch =mnist.train.next_batch(N)
    Y_batch =np.argmax(Y_batch,axis=1)#取出array中元素最大值所对应的索引,二维矩阵，axis=0 按列，axis=1 按行
    info_dict =inference.update(feed_dict={x:X_batch,y_ph: Y_batch})
    inference.print_progress(info_dict)

#save weights and bias
save_w_b(qW_0,qW_1,qb_0,qb_1)
#reload the weights and bias, to verity whether the saved file is OK
qW_0_reload,qW_1_reload,qb_0_reload,qb_1_reload =load_w_b()
X_test =mnist.test.images
print("###########predicting############")
Y_test =np.argmax(mnist.test.labels,axis=1)

n_samples =20
prob_lst =[]
samples =[]
w_samples =[]
b_samples =[]
for _ in range(n_samples):
    """
    w_samp_0 = qW_0.sample()
    w_samp_1 = qW_1.sample()
    b_samp_0 = qb_0.sample()
    b_samp_1 = qb_1.sample()
    """
    w_samp_0 = qW_0_reload.sample()
    w_samp_1 = qW_1_reload.sample()
    b_samp_0 = qb_0_reload.sample()
    b_samp_1 = qb_1_reload.sample()
    
    #w_samp = qw.sample()
    #b_samp = qb.sample()
    #w_samples.append(w_samp)
    #b_samples.append(b_samp)

    prob = tf.nn.softmax(two_layers_NN(X_test, w_samp_0, w_samp_1,b_samp_0, b_samp_1))
    #prob =tf.nn.softmax(tf.matmul(X_test,w_samp)+b_samp)
    prob_lst.append(prob.eval())
    #对于一个二维矩阵，第0个维度代表最外层方括号所框下的子集，
    #第1个维度代表内部方括号所框下的子集。维度越高，括号越小
    #低维拼接等于拿掉最外面括号，高维拼接是拿掉里面的括号(保证其他维度不变)
    #axis=-1表示倒数第一个维度
    # reshape(t, [-1]) == >pass '[-1]' to flatten 't'

    #sample =tf.concat([tf.reshape(w_samp,[-1]),b_samp],0)#数组拼接
    #samples.append(sample.eval())
accy_test =[]
print("lable_predict:")
for prob in prob_lst:
    y_trn_prd =np.argmax(prob,axis=1).astype(np.float32)
    print(y_trn_prd)
    acc =(y_trn_prd ==Y_test).mean()*100
    #print(acc)
    accy_test.append(acc)
plt.hist(accy_test)
plt.title("Histogram of prediction accuracies in the MNIST test data")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.show()

Y_pred =np.argmax(np.mean(prob_lst,axis=0),axis=1)
print("accuracy in predicting the test data =",(Y_pred==Y_test).mean()*100)
