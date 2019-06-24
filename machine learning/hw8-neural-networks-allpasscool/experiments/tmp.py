import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from .load_data import load_mnist_data,_load_mnist
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(28*28,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

data_size = [500, 1000, 1500, 2000]
times = []
accuracy = []
test_features, test_targets = _load_mnist('data','testing')
test_features = test_features.reshape(-1,28*28)
test_features = torch.from_numpy(test_features)
test_targets = torch.from_numpy(test_targets)
test_targets = test_targets.long()
test_dataset = Data.TensorDataset(test_features,test_targets)
test_loader = Data.DataLoader(test_dataset,batch_size=10,shuffle=False,num_workers=2)
for ds in data_size:
    net = Net()
    net = net.float()
    criteration = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=0.01)

    train_features1, test_features1, train_targets1, test_targets1 = load_mnist_data(10,1.0,ds//10,'.')
    train_features1 = torch.from_numpy(train_features1).float()
    train_targets1 = torch.from_numpy(train_targets1).float()
    train_targets1 = train_targets1.long()
    train_dataset1 = Data.TensorDataset(train_features1,train_targets1)
    train_loader1 = Data.DataLoader(train_dataset1,batch_size=10,shuffle=True,num_workers=2)
    
    start = time.time()
    for epoch in range(100):
        runing_loss = 0.0
        for i, data in enumerate(train_loader1,0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            
            loss = criteration(outputs, labels)
            loss.backward()
            optimizer.step()
    end = time.time()
    times.append(end-start)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy.append(correct / total)

print(times)
plt.plot(data_size,times)
plt.title('Running Time')
plt.xlabel('Training Examples')
plt.ylabel('Training Time')
plt.show()

plt.plot(data_size,accuracy)
plt.title('Accuracy')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.show()