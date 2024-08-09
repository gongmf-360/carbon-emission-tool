import numpy as np
import torch
import pandas as pd
import configparser
import random

"""
电力：100 kWh    天然气：50 m³
铝：20 kg        钢：30 kg

电力：0.5 kg CO₂e/kWh            天然气：2.75 kg CO₂e/m³
铝：8.0 kg CO₂e/kg                钢：1.9 kg CO₂e/kg

"""

total_record = 1000

# 生成[50,300）的随机矩阵，1000行2列
rand_arr_1 = torch.randint(50, 300, (total_record, 2))

# 生成[20,100）的随机矩阵，1000行2列
rand_arr_2 = torch.randint(20, 100, (total_record, 2))

# 生成随机整数列表, 用于增加噪声
random_list = [random.randint(-150, 150) for _ in range(total_record)]
print(random_list)

test_data = torch.cat((rand_arr_1, rand_arr_2), dim=1)
print(test_data)
print(test_data.shape)

config = configparser.ConfigParser()
# 使用 'r' 模式和 utf-8 编码打开文件
with open('coefficient_config.ini', 'r', encoding='utf-8') as configfile:
    config.read_file(configfile)
value = float(config.get('coefficient', 'power'))
value2 = float(config.get('coefficient', 'gas'))
value3 = float(config.get('coefficient', 'aluminum'))
value4 = float(config.get('coefficient', 'steel'))
print("coefficient keys: ",  config.options("coefficient"))
print("value, value2, value3, value4: ", value, value2, value3, value4)

# 创建一个系数Tensor
coefficients_tensor = torch.tensor([value, value2, value3, value4])
carbon_emission_tensor = torch.sum(test_data * coefficients_tensor, dim=1)
# 加上噪声
carbon_emission_tensor = carbon_emission_tensor +  torch.tensor(random_list)
# 使其成为一列，n行
carbon_emission_tensor = carbon_emission_tensor.unsqueeze(1)
test_data = torch.cat((test_data, carbon_emission_tensor), dim=1)
print("乘以对应的系数后的tensor，新增一列：")
print(test_data)

# 将Tensor转换为NumPy数组
numpy_array = test_data.numpy()

# 将NumPy数组转换为Pandas DataFrame
data_frame = pd.DataFrame(numpy_array)
print(data_frame)

# 设置列名
columns = config.options("coefficient")
columns.append('carbon_emission')
data_frame.columns = columns

# 写入CSV文件 , carbon_emission
data_frame.to_csv('green_data.csv', index=False, header=True)
