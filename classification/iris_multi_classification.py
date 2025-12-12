from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import japanize_matplotlib

# 概要
# irisデータセットを用いた多値分類モデルの実装だよ~
# 4つの特徴量からどのあやめかを推測できるようにする。
# 方針
# Linear -> CrossEntropyLoss

# x->あやめの4つの特徴量の値。y->xに対応する正解ラベル(0~3)
x,y = load_iris(return_X_y=True)

# データ収集
# 訓練用とテスト用でデータを分ける。
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=100,test_size=50,random_state=111)
inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).long() # long()だよ。nn.CrossEntropyLossに渡す正解値のindexは整数値でないといけないからね
inputs_test = torch.tensor(x_test).float()
labels_test = torch.tensor(y_test).long()


n_input = 4 # 特徴量の個数(今回は4個)
n_output = 3 # 出力数(今回は3種類のあやめの予測値をそれぞれ出力)

## 今回の予測関数は、単純な線形関数のみ(y=Wx+b的なやつね)
class Model(nn.Module):
	def __init__(self,n_input,n_output):
		super().__init__()
		self.l1 = nn.Linear(n_input,n_output)

	def forward(self,x):
		val = self.l1(x)
		return val

# ハイパーパラメータなど
model = Model(n_input,n_output)
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr)
num_epochs = 10000
history = []

# 学習
for epoch in range(num_epochs):
	optimizer.zero_grad()
	outputs = model(inputs)
	loss = criterion(outputs,labels)
	loss.backward()
	optimizer.step()
	predicted = torch.max(outputs,dim=1)[1] # outputsの中で一番大きな値は、結局nn.CrossEntropyLossを通しても最大なので。
	train_acc = (labels==predicted).sum() / len(labels)
	
	with torch.no_grad():
		outputs_test = model(inputs_test)
		loss_test = criterion(outputs_test,labels_test)
		predicted_test = torch.max(outputs_test,dim=1)[1]
		test_acc = (labels_test==predicted_test).sum() / len(labels_test)
		
	if epoch % 100 == 0:
		print(f"(epoch:{epoch} train_loss:{loss.item()} train_acc:{train_acc} loss_test:{loss_test} test_acc:{test_acc})")
		message = np.array([epoch,loss.item(),train_acc,loss_test.item(),test_acc])
		history.append(message)

history = np.array(history)
# 学習曲線の表示
fig,axes = plt.subplots(1,2)
axes[0].set_title('学習曲線(損失)')
axes[0].plot(history[:,0],history[:,1],label='訓練用',c='b')
axes[0].plot(history[:,0],history[:,3],label='テスト用',c='r')
axes[0].set_xlabel('学習回数')
axes[0].set_ylabel('損失')
axes[0].legend()

axes[1].set_title('学習曲線(精度)')
axes[1].plot(history[:,0],history[:,2],label='訓練用',c='b')
axes[1].plot(history[:,0],history[:,4],label='テスト用',c='r')
axes[1].set_xlabel('学習回数')
axes[1].set_ylabel('精度')
axes[1].legend()

plt.show()


