'''
本程序能构建任意层的神经网络
执行分为两部分：
    第一部分正向传播：
    第二部分反向更新
create by Jiang at 2019.07.08
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def sigmoid(x):
    return np.tanh(x)

# 函数sigmoid的派生函数
def dsigmoid(x):
    return 1.0 - np.multiply(x,x)

def mnist_extraction():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    train_nums = mnist.train.num_examples
    validation_nums = mnist.validation.num_examples
    test_nums = mnist.test.num_examples
    print('MNIST数据集的个数')
    print(' >>>train_nums=%d' % train_nums, '\n',
          '>>>validation_nums=%d' % validation_nums, '\n',
          '>>>test_nums=%d' % test_nums, '\n')

    '''2)获得数据值'''
    train_data = mnist.train.images  # 所有训练数据
    val_data = mnist.validation.images  # (5000,784)
    test_data = mnist.test.images  # (10000,784)
    print('>>>训练集数据大小：', train_data.shape, '\n',
          '>>>一副图像的大小：', train_data[0].shape)
    '''3)获取标签值label=[0,0,...,0,1],是一个1*10的向量'''
    train_labels = mnist.train.labels  # (55000,10)
    val_labels = mnist.validation.labels  # (5000,10)
    test_labels = mnist.test.labels  # (10000,10)

    # print('>>>训练集标签数组大小：', train_labels.shape, '\n',
    #       '>>>一副图像的标签大小：', train_labels[1].shape, '\n',
    #       '>>>一副图像的标签值：', train_labels[0])
    #
    # '''4)批量获取数据和标签【使用next_batch(batch_size)】'''
    # batch_size = 100  # 每次批量训练100幅图像
    # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    # print('使用mnist.train.next_batch(batch_size)批量读取样本\n')
    # print('>>>批量读取100个样本:数据集大小=', batch_xs.shape, '\n',
    #       '>>>批量读取100个样本:标签集大小=', batch_ys.shape)
    return mnist

class NN():

    def __init__(self,layers,mnist):

        L = len(layers)
        # 设置神经元权重参数和偏置参数
        self.W = {}
        self.b = {}

        for l in range(1,L):
            # print(layers[l])
            self.W["W" + str(l)] = np.mat(np.random.randn(layers[l], layers[l - 1]) * 0.1)
            self.b["b" + str(l)] = np.mat(np.random.randn(layers[l],1) * 0.1)

        self.n = {}
        for l in range(0,L):
            self.n["n" + str(l)] = layers[l]

        self.A = {}
        self.Z = {}
        self.cache = {}

        # 设置输入矩阵,输出矩阵
        m = len(pat)
        self.inputs = []
        self.output = []
        self.mnist = mnist
        # for data in train_data:
        #     self.inputs.append(data)
        # for labels in train_labels:
        #     self.output.append(labels)
        # self.inputs = np.mat(self.inputs).transpose()
        # self.output = np.mat(self.output).transpose()
        # print(self.inputs.shape,self.output.shape)

    def forward_activation_02(self,L):
        n,m = self.inputs.shape
        if n != self.n['n0']:
            raise ValueError('与输入层节点数不符')

        # 初始化输入
        self.A["A0"] = self.inputs

        for l in range(1,L):
            self.Z["Z" + str(l)] = self.W["W" + str(l)] * self.A["A" + str(l-1)] + self.b["b" + str(l)]
            self.A["A" + str(l)] = sigmoid(self.Z["Z" + str(l)])

        # 更新cache
        for i in range(1,L):
            l = L-i
            if l == L-1:
                self.cache["C" + str(l)] = np.multiply(self.A["A" + str(l)] - self.output,dsigmoid(self.A["A" + str(l)]))
                # print('asllas',self.cache["C" + str(l)])
            else:
                self.cache["C" + str(l)] = np.multiply(self.W["W" + str(l+1)].T*self.cache["C" + str(l+1)],
                dsigmoid(self.A["A" + str(l)]))
                # pass

        # print(self.A["A"+str(L-1)])
        err = np.sum(np.abs(self.A["A"+str(L-1)]-self.output))
        return 1.0/2*err*err

    def backPropagate_02(self,learning_rate,L):
        alpha = learning_rate
        m = self.inputs.shape[1]

        for i in range(L):
            l = L - i - 1
            if l > 0:
                self.b['b' + str(l)] = self.b['b' + str(l)] - alpha * 1.0/m*(self.cache["C" + str(l)]*np.ones((m,1)))
                self.W['W' + str(l)] = self.W['W' + str(l)] - alpha * 1.0/m*(self.cache["C" + str(l)] * self.A["A" + str(l-1)].T)

    def init_prameter(self,batch_size):
        # 每次批量训练batch_size幅图像
        batch_xs, batch_ys = self.mnist.train.next_batch(batch_size)
        self.inputs = np.mat(batch_xs).transpose()
        self.output = np.mat(batch_ys).transpose()

    def train(self,iterations,learning_rate,L,batch_size):
        # 批处理训练
        m = 100 # m为批处理的次数
        for i in range(iterations):
            for j in range(m):
                self.init_prameter(batch_size)
                err = self.forward_activation_02(L)
                self.backPropagate_02(learning_rate,L)
            if i % 100 == 0:
                print(self.A["A" +str(L-1)])
                print('第%d次迭代误差----%lf'%(i,err))

    def run(self,iterations,learning_rate,L,batch_size):
        self.train(iterations,learning_rate,L,batch_size)

    def test(self,L):
        batch_xs, batch_ys = self.mnist.train.next_batch(100)

        self.inputs = np.mat(batch_xs).transpose()
        self.output = np.mat(batch_ys).transpose()
        ans =[]
        predict_ans = []
        for i in len(self.output):
            k = 0
            for j in self.output[:,i]:
                if j == 1:
                    ans.append(k+1)
                    break
                k += 1
        print("100个测试样本的实际结果：")
        print(ans)
        self.forward_activation_02(L)
        print(self.A["A"+str(L-1)],'\n')

if __name__ == '__main__':

    # 实例，构建一个三层的神经网络处理异或。
    minst = mnist_extraction()
    pat = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]
    #各层网络神经元的参数
    layers = [784,16,10]
    L = len(layers)
    nn = NN(layers,minst)
    nn.run(25000,0.2,L,100)
    nn.test(L)
    # nn.test(np.mat([[0],[1]]),L)