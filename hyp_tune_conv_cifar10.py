
import torch
import torchvision.transforms as T
import torchvision.datasets as Set
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader as D
from torch.utils.data import random_split
import torch.optim as optim
log_interval = 100
device = torch.device("cuda") #run on GPU
Batch_size = 128 # train in batches
epoch = 10   # amount of training iterations before testing

CIFAR10_Train = Set.CIFAR10('/CIFAR10_dataset/', train=True, download=True,
                               transform=T.Compose([T.RandomCrop(32, padding=4),
                               T.RandomHorizontalFlip(),
                               T.ToTensor(),T.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010)),]))
#transform:composes several transforms together
# first arg changes image to tensor
#Normalize???? *************************************************************
CIFAR10_Test = Set.CIFAR10('/CIFAR10_dataset/', train=False, download=True,
                               transform=T.Compose([T.ToTensor(),
                               T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]))
TV_split = [45000, 5000] 
CIFAR10_training_set, CIFAR10_validation_set = random_split(CIFAR10_Train, TV_split) #create validation subset

Training_DataLoader = D(CIFAR10_training_set, batch_size = Batch_size, shuffle = True) #shuffle to randomize
Validation_DataLoader = D(CIFAR10_validation_set, batch_size = Batch_size, shuffle = True)
Test_DataLoader = D(CIFAR10_Test, batch_size = Batch_size, shuffle = True)

class FullyConnectedNet(nn.Module): 
  def __init__(self):
    super(FullyConnectedNet, self).__init__() #Super???************************
    self.FC_Layer_1 = nn.Linear(3*32*32, 10) # 28x28 pixels in, 10 labels out
  
  def forward(self,x):
    x = x.view(x.size(0), -1)##################???????????????????? Flatten
    x = F.relu(self.FC_Layer_1(x)) #Activation Function
    return x

class Multi_FC(nn.Module): 
  def __init__(self):
    super(Multi_FC, self).__init__() #Super???************************
    self.FC_Layer_1 = nn.Linear(3*32*32, 4000) # 28x28 pixels in, 10 labels out
    self.FC_Layer_2 = nn.Linear(4000, 2500) # 28x28 pixels in, 10 labels out
    self.FC_Layer_3 = nn.Linear(2500, 1500)
    self.FC_Layer_4 = nn.Linear(1500, 512)
    self.FC_Layer_5 = nn.Linear(512, 128)
    self.FC_Layer_6 = nn.Linear(128, 10)

  def forward(self,x):
    x = x.view(x.size(0), -1)##################???????????????????? Flatten
    x = F.relu(self.FC_Layer_1(x)) #Activation Function
    x = F.relu(self.FC_Layer_2(x))
    x = F.relu(self.FC_Layer_3(x))
    x = F.relu(self.FC_Layer_4(x))
    x = F.relu(self.FC_Layer_5(x))
    
    x = self.FC_Layer_6(x)
    return x

class conv_net(nn.Module): #batchnorm added
  def __init__(self):
    super(conv_net, self).__init__() #Super???************************
    self.CONV_Layer_1 = nn.Conv2d(in_channels = 3, out_channels=32,        
                                  kernel_size= 5, stride=1, padding=0)
    self.conv_1_BN = nn.BatchNorm2d(32)
    nn.Dropout(0.05)
    nn.init.xavier_uniform(self.CONV_Layer_1.weight)

    self.MAXPOOL_Layer_1 = nn.MaxPool2d(kernel_size = 2, stride = 1)
    nn.Dropout(0.2)

    self.CONV_Layer_2 = nn.Conv2d(in_channels = 32, out_channels=64,        
                                  kernel_size= 4, stride=1, padding=0)
    self.conv_2_BN = nn.BatchNorm2d(64)
    nn.Dropout(0.05)
    nn.init.xavier_uniform(self.CONV_Layer_2.weight)
    
    #32-5+1 = 28 -> 28-4+1 = 25
    self.FC_Layer_1 = nn.Linear(24*24*64, 10) 

  def forward(self,x):
    x = F.relu(self.CONV_Layer_1(x))
    x = F.relu(self.MAXPOOL_Layer_1(x))
    x = F.relu(self.CONV_Layer_2(x))
    x = x.view(x.size(0), -1)##################???????????????????? Flatten
    x = self.FC_Layer_1(x)
    return x

def train_network(epoch):
  setSize = len(Training_DataLoader)
  net_EX.train()
  for batchIndex, (inputData, targetLabel) in enumerate(Training_DataLoader):
    #send input and target to GPU
    inputData = inputData.to(device) 
    targetLabel = targetLabel.to(device)
    SGD_Optimizer.zero_grad() #compute gradient
    output = net_EX(inputData) #get output from the model
    loss = criterion(output,targetLabel) #Cross Entropy Loss
    loss.backward() #Back Propogation
    SGD_Optimizer.step() # Update parameters
    if batchIndex % log_interval == 0:
      print('Training Epoch: {}\n{:.0f}%  Complete\tLoss: {:.6f}'.format(
        epoch, 100. * batchIndex / setSize, loss.item()))

def validate_network(epoch):
  net_EX.eval()
  validation_loss = 0
  correct = 0
  setSize = len(Validation_DataLoader.dataset)
  with torch.no_grad():
    for inputData, targetLabel in (Validation_DataLoader):
      inputData = inputData.to(device)
      targetLabel = targetLabel.to(device)
      output = net_EX(inputData)  #get output from the model
      validation_loss += criterion(output,targetLabel)
      predition_label = output.data.max(1, keepdim=True)[1] #get prediction
      #add correct predictions
      correct += predition_label.eq(targetLabel.data.view_as(predition_label)).sum()

  validation_loss /= setSize
  print('\nValidation set: Training Epoch {}\n Average loss: {:.8f}\n Accuracy: {}/{}= {:.2f}%\n'.format(
    epoch, validation_loss, correct, setSize,
    100. * correct / setSize))

def test_network():
  net_EX.eval()
  validation_loss = 0
  correct = 0
  setSize = len(Test_DataLoader.dataset)
  with torch.no_grad():
    for inputData, targetLabel in (Test_DataLoader):
      inputData = inputData.to(device)
      targetLabel = targetLabel.to(device)
      output = net_EX(inputData)  
      validation_loss += criterion(output,targetLabel)
      predition_label = output.data.max(1, keepdim=True)[1]
      correct += predition_label.eq(targetLabel.data.view_as(predition_label)).sum()

  validation_loss /= setSize
  print('\nTest set: \n Average loss: {:.8f} \n Accuracy: {}/{} ={:.2f}%\n'.format(
    validation_loss, correct, setSize,
    100. * correct / setSize))
  accuracy = correct / setSize

  return (100. * correct / setSize)

momentum = [0.6,0.65,0.7,0.85,0.9,0.95,0.975] #
lr = [0.001,0.003,0.006,0.009,0.0001,0.0003,0.0006,0.0009]   #
list1=[]


for i in lr:
  for j in momentum:
    net_EX = conv_net().to(device)
    learning_rate_SGD = i
    momentum_SGD = j
      #optimize with Stochastic Gradient descent
    SGD_Optimizer = optim.SGD(net_EX.parameters(), 
    lr = learning_rate_SGD, momentum = momentum_SGD)
    criterion = nn.CrossEntropyLoss() 

    validate_network(0)
    for k in range(1,epoch+1):
      train_network(k)
      validate_network(k)
    result = test_network().item()
    list1.append([learning_rate_SGD,momentum_SGD,result])

for i in list1:
  learn = round(i[0],3)
  momen = i[1]
  acc = i[2]
  print('\n learning Rate: {}\n momentum: {} \n Accuracy: {}%'.format(
    learn,momen, acc))
