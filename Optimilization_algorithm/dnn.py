'''
该程序构建一个神经网络基本模型，先不考虑Adam优化
'''

import numpy as np
import nn_utils
import time

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

# 函数sigmoid的派生函数
def dsigmoid(x):
    return np.dot(x,(1-x).T)

def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    return np.maximum(0,x)
    # print(s)

def drelu(x):
    """
    Compute the drelu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

keep_prob = 0.8

class NN():

    def __init__(self,layers,X,Y):

        L = len(layers)
        self.L = L
        # 设置神经元权重参数和偏置参数
        self.W = {}
        self.b = {}

        for l in range(1, L):
            # print(layers[l])
            self.W["W" + str(l)] = np.mat(np.random.randn(layers[l], layers[l - 1]) * 0.1)
            self.b["b" + str(l)] = np.mat(np.random.randn(layers[l], 1) * 0.1)

        self.n = {}
        for l in range(0, L):
            self.n["n" + str(l)] = layers[l]

        self.A = {}
        self.Z = {}
        self.cache = {}
        self.db = {}
        self.dw = {}


        self.X = X
        self.Y = Y
        self.cost_gd = []

    def forward_activation_gd(self,flag):
        '''
        :param L: 神经网络的层数
        :param flag: flag为标记，是否启用dropOut正则化，flag = 1启用dropOut正则化
        如果你熟悉，dropout正则化，可以令flag = 0，关闭它。
        :return: 返回大致误差
        '''
        n,m = self.X.shape
        L = self.L
        if n != self.n['n0']:
            raise ValueError('与输入层节点数不符')

        # 初始化输入,一次性处理全部样本
        self.A["A0"] = self.X
        for l in range(1,L):
            if flag == 0 or l == 1 or l == L-1:
                self.Z["Z" + str(l)] = self.W["W" + str(l)] * self.A["A" + str(l-1)] + self.b["b" + str(l)]
                if l == L-1:
                    self.A["A" + str(l)] = sigmoid(self.Z["Z" + str(l)])
                else:
                    self.A["A" + str(l)] = relu(self.Z["Z" + str(l)])
            else:
                #启用dropout正则化
                # print(self.A["A" + str(l-1)])

                self.d = np.random.rand(self.A["A" + str(l-1)].shape[0],self.A["A" + str(l-1)].shape[1])
                self.d = self.d < keep_prob
                self.A["A" + str(l-1)] = np.multiply(self.A["A" + str(l-1)],self.d)
                self.A["A" + str(l-1)] /= keep_prob
                self.Z["Z" + str(l)] = self.W["W" + str(l)] * self.A["A" + str(l - 1)] + self.b["b" + str(l)]
                self.A["A" + str(l)] = relu(self.Z["Z" + str(l)])
        # print('self.A',self.A)
        # time.sleep(2)  # delays for 5 seconds

    def backPropagate_gd(self):
        m = self.X.shape[1]
        for l in reversed(range(1,self.L)):
            if l == self.L-1:
                self.cache["C" + str(l)] = np.multiply(self.A["A" + str(l)] - self.Y,dsigmoid(self.A["A" + str(l)]))
            else:
                self.cache["C" + str(l)] = np.multiply(self.W["W" + str(l+1)].T*self.cache["C" + str(l+1)],
                                                       drelu(self.A["A" + str(l)]))
            self.db['db' + str(l)] = self.cache["C" + str(l)]*np.ones((m,1))
            self.dw['dw' + str(l)] = self.cache["C" + str(l)]*self.A["A" + str(l-1)].T

    def updata_parameters(self,learning_rate):
        alpha = learning_rate
        m = self.X.shape[1]
        L = self.L
        for l in reversed(range(1,L)):
            self.b['b' + str(l)] = self.b['b' + str(l)] - alpha * 1.0/m*self.db["db"+str(l)]
            self.W['W' + str(l)] = self.W['W' + str(l)] - alpha * 1.0/m*self.dw['dw'+str(l)]

    def init_prameter_gd(self,batch_size,X,Y):
        pass

    def train(self,iterations,learning_rate,batch_size,flag,is_print):
        # 批处理训练
        m = 100 # m为批处理的次数
        L = self.L
        for i in range(iterations):
            for j in range(m):
                # self.init_prameter(batch_size)
                self.forward_activation_gd(flag)
                cost = nn_utils.compute_cost(self.A["A" + str(self.L - 1)],self.Y)
                self.backPropagate_gd()
                self.updata_parameters(learning_rate)
            if i % 100 == 0:
                self.cost_gd.append(cost)
                if i % 1000 == 0 and is_print:
                    print('第%d次迭代误差----%lf'%(i,cost))

    def test(self):
        batch_xs, batch_ys = self.mnist.test.next_batch(100)
        self.inputs = np.mat(batch_xs).transpose()
        self.output = np.mat(batch_ys).transpose()
        self.forward_activation_02(0)
        print("100个测试样本的实际结果：")
        print(self.output.transpose())
        print("100个测试样本的预测结果：")
        print(self.A["A"+str(self.L-1)].transpose())
