import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import torch.optim as optim

## 隠れ層なし+活性化関数なし
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Linear(1,1)
        
    def forward(self,x):
        val = self.layer1(x)
        return val
    
## 隠れ層あり+活性化関数なし
class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Linear(1,10)
        self.layer2 = nn.Linear(10,10)
        self.layer3 = nn.Linear(10,1)
        
    def forward(self,x):
        val = self.layer1(x)
        val = self.layer2(val)
        val = self.layer3(val)
        return val    
    
class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Linear(1,10)
        self.layer2 = nn.Linear(10,10)
        self.ReLU = nn.ReLU()
        self.layer3 = nn.Linear(10,1)
        
    def forward(self,x):
        val = self.layer1(x)
        val = self.layer2(val)
        val = self.ReLU(val)
        val = self.layer3(val)
        return val  
    
# データを集める
x = np.random.rand(100,1)*2 - 1
y = x**2 # y=x^2のグラフを学習させる
x_train = x[:50,:]
x_test = x[50:,:]
y_train = y[:50,:]
y_test = y[50:,:]
inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).float()
inputs_test = torch.tensor(x_test).float() # numpy->Tensor
labels_test = torch.tensor(y_test).float() # numpy->Tensor
lr = 0.05 #学習率
num_epochs = 1000 #エポック数
model1 = Model1()
model2 = Model2()
model3 = Model3()
title1 = '隠れ層なし'
title2 = '隠れ層あり,活性化関数(ReLU)なし'
title3 = '隠れ層あり,活性化関数(ReLU)あり'
fig,axes = plt.subplots(1,3,figsize=(20,8)) # Figure->全体 axes->1つ1つのプロットを描く領域。subplotsは、fig = plt.figure() fig.add_subplot()と同じ。
fig.suptitle('y=x^2を学習させた結果')

def train_model(model,ax,title):
    # 最適化関数として確率的勾配降下法(SGD)を使用
    optimizer = optim.SGD(model.parameters(),lr=lr)
    ## 損失関数として平均二乗誤差(MSE)を使用
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad() #勾配を初期化
        outputs = model(inputs) #予測出力
        loss = criterion(outputs,labels) #損失
        loss.backward() #逆伝播
        optimizer.step() #パラメータ(W,B)を調整
    
    with torch.no_grad():
        labels_pred = model(inputs_test)
    
    ax.scatter(inputs_test.numpy(),labels_test.numpy(),c='k',label='正解データ')   
    ax.scatter(inputs_test.numpy(),labels_pred.numpy(),c='b',marker='x',label='予測データ')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    
## 学習するよ
train_model(model1,axes[0],title1)
train_model(model2,axes[1],title2)
train_model(model3,axes[2],title3)

# グラフにするよ
plt.show()