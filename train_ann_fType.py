# 不同类别特征权重测试
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
from datetime import datetime
from ANNmodel import ANN
from sklearn.preprocessing import normalize

INPUT_SIZE = 404    # 输入维度
HIDDEN_SIZE1 = 256  # 隐藏层1输入维度
HIDDEN_SIZE2 = 64   # 隐藏层2输入维度
HIDDEN_SIZE3 = 32   # 隐藏层2输出维度
OUTPUT_SIZE = 3      # 输出维度
BATCH_SIZE = 100      # batch size
EPOCH = 50          # 训练轮次
LR = 0.001            # 学习率

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据融合
data_x1 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\data_400\ANNData_xf_400.csv', sep=',', header=None)
data_y1 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\data_400\ANNData_yf.csv', sep=',', header=None)

data_x2 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\data_400\ANNData_lean_400.csv', sep=',', header=None)
data_y2 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\data_400\ANNData_lean_y.csv', sep=',', header=None)

data_x3 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\data_400\ANNData_disturb_400.csv', sep=',', header=None)
data_y3 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\data_400\ANNData_disturb_y.csv', sep=',', header=None)

feature_x1 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\feature_expend\ANNData_xf_feature.csv', sep=',', header=None)
feature_x2 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\feature_expend\ANNData_lean_feature.csv', sep=',', header=None)
feature_x3 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\feature_expend\ANNData_disturb_feature.csv', sep=',', header=None)

feature_x1 = pd.DataFrame(feature_x1)
feature_x2 = pd.DataFrame(feature_x2)
feature_x3 = pd.DataFrame(feature_x3)

feature_x11 = feature_x1.iloc[:, 16:20]
feature_x21 = feature_x2.iloc[:, 16:20]
feature_x31 = feature_x3.iloc[:, 16:20]

data_x1 = np.append(data_x1, feature_x11, axis=1)
data_x2 = np.append(data_x2, feature_x21, axis=1)
data_x3 = np.append(data_x3, feature_x31, axis=1)

# 数据分割
data_x2 = pd.DataFrame(data_x2)                   # 转化为dataframe格式，才能使用iloc分割矩阵
data_y2 = pd.DataFrame(data_y2)
data_x3 = pd.DataFrame(data_x3)                   # 转化为dataframe格式，才能使用iloc分割矩阵
data_y3 = pd.DataFrame(data_y3)

data_x21 = data_x2.iloc[0:400, :]
data_y21 = data_y2.iloc[0:400]
data_x31 = data_x3.iloc[0:400, :]
data_y31 = data_y3.iloc[0:400]

data_x21 = np.array(data_x21)
data_y21 = np.array(data_y21)
data_x31 = np.array(data_x31)
data_y31 = np.array(data_y31)

data_x = np.append(data_x1, data_x21, axis=0)
data_y = np.append(data_y1, data_y21, axis=0)
data_x = np.append(data_x, data_x31, axis=0)
data_y = np.append(data_y, data_y31, axis=0)

# data_x = normalize(data_x, axis=1, norm='max')  # 归一化
data_x = pd.DataFrame(data_x)                   # 转化为dataframe格式，才能使用iloc分割矩阵
data_y = pd.DataFrame(data_y)

x_train = data_x.iloc[0:2800, :]                  # 分割训练集和测试集
y_train = data_y.iloc[0:2800]

y_train = y_train.squeeze(1)

x_train = torch.from_numpy(x_train.values).float()  # 转换为tensor格式
y_train = torch.from_numpy(y_train.values).type(torch.LongTensor)

train = torch.utils.data.TensorDataset(x_train, y_train)

train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

net = ANN(INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE)
MSELoss = nn.MSELoss(reduction='mean')
CrossEntropyLoss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[EPOCH//2, EPOCH//4*3], gamma=0.1)

total_loss = []
for epoch in range(EPOCH):
    train_loss = []
    net.train()
    for j, (t_x, t_y) in enumerate(train_loader):
        train = Variable(t_x)
        label = Variable(t_y)

        optimizer.zero_grad()

        predict = net(train)
        loss = CrossEntropyLoss(predict, label)

        loss.backward()
        optimizer.step()

        # step_scheduler.step()
        train_loss.append(loss.item())

        # print(predict)
        # print(label)

    total_loss.append(np.mean(train_loss))
    log_string = 'iter: [{:d}/{:d}], train_loss: {:0.6f}, lr: {:0.7f}'.format((epoch + 1), EPOCH, total_loss[-1], optimizer.param_groups[0]['lr'])
    print(log_string)

with open("ANN_train_loss.txt", 'w') as train_los:
    train_los.write(str(total_loss))

torch.save(net.state_dict(), 'ANN_model.pth')
