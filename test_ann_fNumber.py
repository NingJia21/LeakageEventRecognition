import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
import csv
from datetime import datetime
from ANNmodel import ANN
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix

INPUT_SIZE = 405    # 输入维度
HIDDEN_SIZE1 = 256  # 隐藏层1输入维度
HIDDEN_SIZE2 = 64   # 隐藏层2输入维度
HIDDEN_SIZE3 = 32   # 隐藏层2输出维度
OUTPUT_SIZE = 3      # 输出维度
BATCH_SIZE = 400      # batch size
EPOCH = 100          # 训练轮次
LR = 0.01            # 学习率

data_x1 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\data_400\ANNData_xf_test_400.csv', sep=',', header=None)
data_y1 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\data_400\ANNData_yf_test.csv', sep=',', header=None)

data_x2 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\data_400\ANNData_lean_test_400.csv', sep=',', header=None)
data_y2 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\data_400\ANNData_lean_test_y.csv', sep=',', header=None)

data_x3 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\data_400\ANNData_disturb_test_400.csv', sep=',', header=None)
data_y3 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\data_400\ANNData_disturb_test_y.csv', sep=',', header=None)

data_x4 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\data_400\test_400.csv', sep=',', header=None)
data_y4 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\feature_expend\test_y.csv', sep=',', header=None)

feature_x1 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\feature_expend\ANNData_xf_test_feature.csv', sep=',', header=None)
feature_x2 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\feature_expend\ANNData_lean_test_feature.csv', sep=',', header=None)
feature_x3 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\feature_expend\ANNData_disturb_test_feature.csv', sep=',', header=None)
feature_x4 = pd.read_csv(r'D:\phi_OTDR\3_TrainData_feature_test\feature_expend\test_feature.csv', sep=',', header=None)

feature_x4 = pd.DataFrame(feature_x4)
feature_x41 = feature_x4.iloc[:, 0:2]
feature_x42 = feature_x4.iloc[:, 8:10]
feature_x43 = feature_x4.iloc[:, 16:17]
feature_x41 = np.array(feature_x41)
feature_x42 = np.array(feature_x42)
feature_x43 = np.array(feature_x43)
feature_x4 = np.append(feature_x41, feature_x42, axis=1)
feature_x4 = np.append(feature_x4, feature_x43, axis=1)

data_x1 = np.append(data_x1, feature_x1, axis=1)
data_x2 = np.append(data_x2, feature_x2, axis=1)
data_x3 = np.append(data_x3, feature_x3, axis=1)
data_x4 = np.append(data_x4, feature_x4, axis=1)

# data_x = pd.read_csv(r'D:\phi_OTDR\3_TrainData\ANNData_Gnoise.csv', sep=',', header=None)
# data_y = pd.read_csv(r'D:\phi_OTDR\3_TrainData\ANNData_otherevents_y.csv', sep=',', header=None)

data_x1 = pd.DataFrame(data_x1)                   # 转化为dataframe格式，才能使用iloc分割矩阵
data_y1 = pd.DataFrame(data_y1)
data_x11 = data_x1.iloc[0:50, :]
data_y11 = data_y1.iloc[0:50]
data_x12 = data_x1.iloc[50:250, :]
data_y12 = data_y1.iloc[50:250]
data_x11 = np.array(data_x11)
data_y11 = np.array(data_y11)
data_x12 = np.array(data_x12)
data_y12 = np.array(data_y12)

data_x = np.append(data_x11, data_x2, axis=0)
data_y = np.append(data_y11, data_y2, axis=0)
data_x = np.append(data_x, data_x12, axis=0)
data_y = np.append(data_y, data_y12, axis=0)
data_x = np.append(data_x, data_x3, axis=0)
data_y = np.append(data_y, data_y3, axis=0)

# data_x = normalize(data_x, axis=1, norm='max')  # 归一化

# data_x = pd.DataFrame(data_x)
# data_y = pd.DataFrame(data_y)
data_x4 = pd.DataFrame(data_x4)
data_y4 = pd.DataFrame(data_y4)

x_test = data_x4.iloc[0:400, :]
y_test = data_y4.iloc[0:400]

y_test = y_test.squeeze(1)

x_test = torch.from_numpy(x_test.values).type(torch.Tensor)
y_test = torch.from_numpy(y_test.values).type(torch.LongTensor)

test = torch.utils.data.TensorDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

net = ANN(INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE)
net.load_state_dict(torch.load("ANN_model.pth"))

'''checkpoint = torch.load('ANN_model.pth', map_location='cpu')
for key in checkpoint:
    print(key)'''

total = 0
correct = 0
total1 = 0
total2 = 0
total3 = 0
net.eval()
for j, (x_test, y_test) in enumerate(test_loader):
    x_test = Variable(x_test)
    output = net(x_test)

    predict = torch.max(output.data, 1)[1]

    print(x_test)
    print(output)
    print(predict)
    print(y_test)

    total += len(y_test)
    correct += (predict == y_test).sum()
    total1 += ((predict == y_test) & (y_test.data == 0)).sum()
    total2 += ((predict == y_test) & (y_test.data == 1)).sum()
    total3 += ((predict == y_test) & (y_test.data == 2)).sum()
    cm = confusion_matrix(y_test, predict)
    print(cm)

    with open("ANN_result.csv", "a", encoding="utf-8", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(predict)
        file.close()
accuracy = 100*correct/float(total)
print(accuracy)
print(total1)
print(total2)
print(total3)


