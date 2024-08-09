import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.init as init

import configparser
import warnings
import torch.utils.data as Data
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")



#数据读取
datas = pd.read_csv('green_data.csv')

# 查看数据
print(datas.head())
print("数据维度", datas.shape)

config = configparser.ConfigParser()
with open('coefficient_config.ini', 'r', encoding='utf-8') as configfile:
    config.read_file(configfile)
feature_columns = config.options("coefficient")

# 特征集
features = datas[feature_columns].values.tolist()

#  结果集
carbon_emission = datas[['carbon_emission']].values.tolist()

# 数据切分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    features, carbon_emission, test_size=0.3, random_state=42)


print("-------------")
print("X_train", X_train)
print("X_test", X_test)
print("y_train", y_train)
print("y_test", y_test)
print("-------------")

# 数据标准化处理scale=StandardScaler()
scale=StandardScaler()
X_train_s = scale.fit_transform(X_train)
X_test_s = scale.transform(X_test)


## 将训练集数据处理为数据表，方便探索数据情况
feature_df = pd.DataFrame(data=X_train_s, columns=feature_columns)
feature_df["target"] = y_train
feature_df.head()


# 设置字体
plt.rcParams['font.family'] = 'SimHei'  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 可视化每个变量的数据分布，密度图
# plt.figure(figsize=(45,35))
# for ii in range(len(feature_columns)):
#     plt.subplot(2, 2, ii + 1)
#     sns.kdeplot(feature_df[feature_columns[ii]], gridsize=200)
#     plt.title(feature_columns[ii])
# plt.subplots_adjust(hspace=0.35)
# plt.show()


# 将数据集转化为张量
train_xt = torch.from_numpy(X_train_s.astype(np.float32))  # np.ndarray (got list)
train_yt = torch.from_numpy(np.array(y_train))
test_xt = torch.from_numpy(X_test_s.astype(np.float32))
test_yt = torch.from_numpy(np.array(y_test))

# 将训练数据为数据加载器
# 转成float类型，否则报错：Found dtype Double but expected Float
train_data = Data.TensorDataset(train_xt.float(), train_yt.float())
# test_data = Data.TensorDataset(test_xt, test_yt)
train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)
#test loader= Data,DataLoader(dataset = test data, batch size=64,
#                      shuffle=True,num workers=1)

#检查训练数据集的一个batch的样本的维度是否正确
print(train_loader)
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
## 输出训练图像的尺寸和标签的尺寸,和数据类型
print("b x.shape:", b_x.shape)
print("b y.shape:", b_y.shape)
print("b x.dtype:", b_x.dtype)
print("b y.dtype:", b_y.dtype)

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



## 创建模型对象，输出我们的网络结构
mlp_module = MultivariateLinearRegression(len(X_train[0]))
# mlp_module = MLPregression(len(X_train[0]))
print("定义的模型 mlp_module：", mlp_module)


# 定义优化器
optimizer = torch.optim.SGD(mlp_module.parameters(), lr=0.001)
# optimizer = torch.optim.Adam(mlp_module.parameters(), lr=0.001)

loss_func = nn.MSELoss()

train_loss_all = []
epoch = 5000
## 对模型进行迭代训练， 对所有的数据训练
for i in range(epoch):
    train_loss = 0
    train_num = 0
    ## 对训练数据迭代器进行迭代计算
    for step, (b_x, b_y) in enumerate(train_loader):
        # 前向传播
        output = mlp_module(b_x)
        # 计算损失
        loss = loss_func(output, b_y)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss_all.append(loss.mean().item())
    if i % 100 == 0:
        print(i, 'Loss: ', loss.mean().item())

## 可视化损失函数的变化情况
plt.figure(figsize=(10,6))
plt.plot(train_loss_all, "ro-", label="Train Loss")
plt.legend()
plt.grid()
plt.xlabel("epoch", size=13)
plt.ylabel("Loss", size=13)
plt.show()


# 保存模型的权重state_dict
try:
    torch.save(mlp_module.state_dict(), 'model_weights/MLP_linear_weights')
except Exception as e:
    print(f"An error occurred while saving the model: {e}")



# 或者加载state_dict，再对对测试集进行预测
mlp_module.load_state_dict(torch.load('model_weights/MLP_linear_weights', map_location=torch.device('cpu')))
mlp_module.eval()  # 切换到评估模式
pre_y = mlp_module(test_xt)

pre_y = pre_y.data.numpy()
print("y_test： ", y_test[0:10])
print("pre_y： ", pre_y[0:10])
mae = mean_absolute_error(y_test, pre_y)
print("在测试集上的平均绝对误差为：", mae)

## 可视化在测试集上真实值和预测值的差异 - 对比
index = np.argsort(y_test)
plt.figure(figsize=(12, 5))
plt.plot(np.arange(len(y_test)), y_test, "r", label="真实值")
plt.scatter(np.arange(len(pre_y)), pre_y, s=3, c="b", label="预测值")
plt.legend(loc = "upper left")
plt.xlabel("Index")
plt.ylabel("Y")
plt.show()


# 可视化在测试集上真实值和预测值的差异 - 散点图
# 创建一个图形和一个子图
fig, ax = plt.subplots()

# 设置y轴标签，使用蓝色字体
ax.set_ylabel('预测值', color='blue')
# 绘制散点图，设置点的大小和形状（这里使用圆点）
scatter = ax.scatter(y_test, pre_y, s=30, marker='o', c='blue')  # 's' 控制点的大小，'marker' 控制点的形状
plt.title('预测值与真实值的散点图', color='red')
plt.xlabel('真实值')
# 绘制一条完美的预测线 (y=x)
lims = [
    np.min([plt.xlim(), plt.ylim()]),  # min of both axes
    np.max([plt.xlim(), plt.ylim()]),  # max of both axes
]
plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
plt.xlim(lims)
plt.ylim(lims)
# 显示图形
plt.show()

