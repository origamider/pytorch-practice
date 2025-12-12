from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import japanize_matplotlib
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# 概要
# MNISTデータセットを用いた数字認識モデルの実装。

# データ準備
data_root = '../data/'
batch_size = 500 # バッチサイズ
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(0.5,0.5),
	transforms.Lambda(lambda x: x.view(-1))
])
train_set = datasets.MNIST(
	root=data_root,
	train=True,
	download=True,
	transform=transform
)
test_set = datasets.MNIST(
	root=data_root,
	train=False,
	download=True,
	transform=transform
)
train_loader = DataLoader(
	dataset=train_set,
	batch_size=batch_size,
	shuffle=True
)
test_loader = DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False
)
class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input,n_hidden)
        self.l2 = nn.Linear(n_hidden,n_output)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        val = self.l1(x)
        val = self.relu(val)
        val = self.l2(val)
        return val
    
torch.manual_seed(1111)
torch.mps.manual_seed(1111)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
num_epochs = 10
lr = 0.1
n_input = 784 # ここには特徴量を入れたい。今回は1枚の画像に28*28のピクセルデータが保存されているので、784で。
n_hidden = 128 # 自由！
n_output = 10 # 正解の選択肢が0~9なので、10通り。
model = Net(n_input,n_hidden,n_output).to(device) # 忘れずにmodelもGPUに移動させてね⭐️
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=lr)
n_train = len(train_set) # 訓練用データの個数
n_test = len(test_set) # テスト用データの個数
history = [] # 結果一覧
# 学習するよ
for epoch in range(num_epochs):
    train_loss,eval_loss = 0,0
    train_acc,eval_acc = 0,0
    for inputs,labels in train_loader:
        train_batch_size = len(labels) # 一見batch_sizeと一緒じゃねって思うが、例えば、n_data=98,batch_size=10の時、最後はbatch_size=8となって取り出される。
        inputs = inputs.to(device) # GPUへデータを移動！
        labels = labels.to(device) # GPUへデータを移動！
        
        optimizer.zero_grad() # 勾配りセット！
        outputs = model(inputs) # 予測値を求める
        loss = criterion(outputs,labels) # 損失を求める
        loss.backward() # 逆伝播
        optimizer.step() # パラメータ(W,B)更新
        
        # 精度確認
        predicted = torch.max(outputs,dim=1)[1]
        train_loss += loss.item() * train_batch_size # ここでのlossはbatchにおける平均値であるため、batch_size分加算する。
        train_acc += (predicted==labels).sum().item()
        
    for inputs_test,labels_test in test_loader:
        test_batch_size = len(labels_test)
        inputs_test = inputs_test.to(device) # GPUへデータを移動！
        labels_test = labels_test.to(device) # GPUへデータを移動！
        
        outputs_test = model(inputs_test)
        loss = criterion(outputs_test,labels_test)
        
        predicted_test = torch.max(outputs_test,dim=1)[1]
        eval_loss += loss.item() * test_batch_size
        eval_acc += (predicted_test==labels_test).sum().item()
        
    train_loss /= n_train
    eval_loss /= n_test
    train_acc /= n_train
    eval_acc /= n_test
    print(f'Epoch : {epoch} train_loss : {train_loss} train_acc : {train_acc} eval_loss : {eval_loss} eval_acc : {eval_acc}')
    message = np.array([epoch,train_loss,train_acc,eval_loss,eval_acc])
    history.append(message)
    
# 学習曲線の表示
history = np.array(history)
fig,axes = plt.subplots(1,2)
axes[0].set_title('学習曲線(損失)')
axes[0].plot(history[:,0],history[:,1],c='r',label='訓練用')
axes[0].plot(history[:,0],history[:,3],c='b',label='テスト用')
axes[0].set_xlabel('繰り返し数')
axes[0].set_ylabel('損失')
axes[0].legend()

axes[1].set_title('学習曲線(損失)')
axes[1].plot(history[:,0],history[:,2],c='r',label='訓練用')
axes[1].plot(history[:,0],history[:,4],c='b',label='テスト用')
axes[1].set_xlabel('繰り返し数')
axes[1].set_ylabel('精度')
axes[1].legend()

plt.show()


# イメージで理解

for images,labels in test_loader:
    break;

inputs = images.to(device)
labels = labels.to(device)
outputs = model(inputs)
predicted = torch.max(outputs,1)[1] # 返り値は(max値,maxに相当するidx)

plt.figure()

for i in range(50):
    ax = plt.subplot(5,10,i+1)
    image = inputs[i]
    label = labels[i]
    pred = predicted[i]
    
    if pred==label:
        c = 'r'
    else:
        c = 'b'
    
    image = (image+1)/2 # [-1,1] -> [0,1]変換
    
    plt.imshow(image.to('cpu').detach().numpy().reshape(28,28),cmap='gray')
    ax.set_title(f'予想:{pred} 正解:{label}',c=c)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()