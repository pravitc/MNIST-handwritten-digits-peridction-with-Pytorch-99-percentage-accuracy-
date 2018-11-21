from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
train_on_gpu = torch.cuda.is_available()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

train_data = datasets.MNIST('data',train=True,transform =transform,download =True)
test_data = datasets.MNIST('data',train=False,download = True,transform =transform)

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(0.2 * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
train_sample = SubsetRandomSampler(train_idx)
valid_sample = SubsetRandomSampler(valid_idx)
#train_loader = torch.utils.data.DataLoader(train_idx,batch_size = 20,sampler = train_sample,num_workers=0)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=20,sampler=train_sample, num_workers=0)
    
valid_loader = torch.utils.data.DataLoader(train_data,batch_size = 20,sampler=valid_sample,num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data,batch_size = 20)

     

class network(nn.Module):
    def __init__(self):
        super(network,self).__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding = 1)
        self.conv2 = nn.Conv2d(16,32,3,padding= 1)
        self.maxpool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(7*7*32,500)
        self.fc2 = nn.Linear(500,10)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self,x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1,7*7*32)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
model = network()
if train_on_gpu:
    model.cuda()
criterion = nn.CrossEntropyLoss()
optmzr = torch.optim.SGD(model.parameters(),lr = 0.01)

epochs = 20
valid_min_loss = np.Inf
for epoch in range(epochs):
    train_loss = 0
    valid_loss = 0
    model.train()
    for images,labels in train_loader:
        if train_on_gpu:
            images,labels = images.cuda(),labels.cuda()  
        optmzr.zero_grad()
        ps = model(images)
        loss = criterion(ps,labels)
        loss.backward()
        optmzr.step()
        train_loss += loss.item()*images.shape[0]
    
    model.eval()
    for images,labels in valid_loader:
        if train_on_gpu:
            images,labels = images.cuda(),labels.cuda()
        output = model(images)
        loss = criterion(output,labels)
        valid_loss += loss.item()*images.shape[0]
        
    train_loss = train_loss /len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    if valid_loss < valid_min_loss:
        valid_min_loss = valid_loss
        torch.save(model.state_dict,'model_mnist.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_min_loss,valid_loss))
        
        
        
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
for images,labels in test_loader:
    if train_on_gpu:
        images,labels = images.cuda(),labels.cuda()
    pred= model(images)
    loss = criterion(pred,labels)
    test_loss += loss.item()*images.shape[0]
    _,pred_tensor = torch.max(pred,1)
    correct_tensor = pred_tensor.eq(labels.data.view_as(pred_tensor))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    for i in range(20):
        label = labels.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))
for i in range(10):
    if class_total[i] > 0:
       # print('Test Accuracy of %5s: %2d% (%2d/%2d)' % (
      #      classes[i], 100 * class_correct[i] / class_total[i],
       #     np.sum(class_correct[i]), np.sum(class_total[i])))
        print("value of accuracy =", 100 * class_correct[i] / class_total[i])


print('\nTest Accuracy (Overall): %02d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
print('total = ', 100. * np.sum(class_correct) / np.sum(class_total))
    
        
       
        
        
        
        

