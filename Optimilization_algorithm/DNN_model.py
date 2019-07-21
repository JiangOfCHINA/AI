import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

import dnn
import nn_utils

def model(X,Y,layers_dims,optimizer,flag,learning_rate=0.0007,
          mini_batch_size=64,beta=0.9,beta1=0.9,beta2=0.999,
          epsilon=1e-8,num_epochs=10000,print_cost=True,is_plot=True):

    """
    可以运行在不同优化器模式下的3层神经网络模型。

    参数：
        X - 输入数据，维度为（2，输入的数据集里面样本数量）
        Y - 与X对应的标签
        flag -是否启用dropout正则化
        layers_dims - 包含层数和节点数量的列表
        optimizer - 字符串类型的参数，用于选择优化类型，【 "gd" | "momentum" | "adam" 】
        learning_rate - 学习率
        mini_batch_size - 每个小批量数据集的大小
        beta - 用于动量优化的一个超参数
        beta1 - 用于计算梯度后的指数衰减的估计的超参数
        beta1 - 用于计算平方梯度后的指数衰减的估计的超参数
        epsilon - 用于在Adam中避免除零操作的超参数，一般不更改
        num_epochs - 整个训练集的遍历次数，（视频2.9学习率衰减，1分55秒处，视频中称作“代”）,相当于之前的num_iteration
        print_cost - 是否打印误差值，每遍历1000次数据集打印一次，但是每100次记录一个误差值，又称每1000代打印一次
        is_plot - 是否绘制出曲线图
    返回：
        parameters - 包含了学习后的参数

    """
    L = len(layers_dims)
    costs = []
    t = 0 #每学习完一个minibatch就增加1
    seed = 10 #随机种子

    #实例化一个神经网络
    nn  = dnn.NN(layers_dims,X,Y)
    #选择优化器
    if optimizer == "gd":
        pass #不使用任何优化器，直接使用梯度下降法
    elif optimizer == "momentum":
        # v = initialize_velocity(parameters) #使用动量
        pass
    elif optimizer == "adam":
        # v, s = initialize_adam(parameters)#使用Adam优化
        pass
    else:
        print("optimizer参数错误，程序退出。")
        exit(1)

    # 开始训练
    # iterations,learning_rate,batch_size,flag,is_print
    # learning_rate = 0.0007,
    # mini_batch_size = 64, beta = 0.9, beta1 = 0.9, beta2 = 0.999,
    # epsilon = 1e-8, num_epochs = 10000, print_cost = True, is_plot = True, flag
    nn.train(num_epochs,learning_rate,mini_batch_size,flag,is_plot)

    #是否绘制曲线图
    if is_plot:
        plt.plot(nn.cost_gd)
        plt.ylabel('cost')
        plt.xlabel('epochs (per 100)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

    return nn.self.W,nn.self.b


if __name__ == '__main__':

    X,Y = nn_utils.load_dataset()
    X = np.mat(X)
    Y = np.mat(Y)

    layers_dims = [2,7,8,1]
    model(X,Y,layers_dims,'gd',1)
