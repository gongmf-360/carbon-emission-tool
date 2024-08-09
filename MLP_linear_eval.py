import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.init as init


import warnings
import torch.utils.data as Data

warnings.filterwarnings("ignore")



#定义MLP模型
class MLPregression(nn.Module):

    def __init__(self, input_dim):
        super(MLPregression, self).__init__()
        hidden_size_1 = 50
        hidden_size_2 = 60
        hidden_size_3 = 100
        output_size = 1
        ## 定义一个隐藏层
        self.hidden1 = nn.Linear(in_features=input_dim, out_features=hidden_size_1, bias=True)
        ## 定义第二个隐藏层
        self.hidden2 = nn.Linear(hidden_size_1, hidden_size_2)
        # ## 定义第三那个隐藏层
        # self.hidden3 = nn.Linear(hidden_size_2, hidden_size_3)
        ## 回归预测层
        self.predict = nn.Linear(hidden_size_2, output_size)
        ## 初始化权重
        init.xavier_uniform_(self.hidden1.weight)
        init.xavier_uniform_(self.hidden2.weight)


    ## 网络的正向传播
    def forward(self, x):
        x = torch.sigmoid(self.hidden1(x))
        # x = torch.sigmoid(self.hidden2(x))
        # x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        output = self.predict(x)
        ## 输出一维的结果
        return output[:, 0]


# 定义多元线性回归模型
class MultivariateLinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(MultivariateLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)



def eval_carbon_emission(X_test):
    print("eval_carbon_emission(X_test): ", X_test)

    # 数据标准化处理scale=StandardScaler()
    scale = StandardScaler()
    # 不加这行会报错：sklearn.exceptions.NotFittedError: This StandardScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
    scale.fit(np.array(X_test).reshape(-1, 1))
    X_test_s = scale.transform(np.array(X_test).reshape(-1, 1))

    # 将数据集转化为张量
    test_xt = torch.from_numpy(X_test_s.astype(np.float32))
    print("test_xt: ", test_xt)
    test_xt = test_xt.reshape(1, -1)
    print("test_xt: ", test_xt)
    ## 输出我们的网络结构
    mlp_module = MultivariateLinearRegression(len(X_test))
    # mlp_module = MLPregression()
    print("定义的模型 mlp_module：", mlp_module)


    # 或者加载state_dict，再对对测试集进行预测
    mlp_module.load_state_dict(torch.load('model_weights/MLP_linear_weights', map_location=torch.device('cpu')))
    mlp_module.eval()  # 切换到评估模式
    pre_y = mlp_module(test_xt)

    pre_y = pre_y.data.numpy()
    print("pre_y： ", pre_y[0:10])
    return pre_y[0]


