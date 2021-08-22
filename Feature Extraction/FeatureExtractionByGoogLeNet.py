
### Training ###

import torch
import torchvision
#import skimage.io as io
import numpy as np
import torchvision.transforms as t
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torchvision.models as model
torch.cuda.set_device(0)

# Data paths
train_path="/content/drive/MyDrive/Sars-cov-2/Train"
#test_path="/content/drive/MyDrive/Sars-cov-2/Test"
val_path="/content/drive/MyDrive/Sars-cov-2/Validation"
plot_path="/content/drive/MyDrive/Sars-cov-2"
snapshot_path="/content/drive/MyDrive/Sars-cov-2"

#Augmentation
model_name='GoogLeNet'
batch_s = 25
transform=t.Compose([t.Resize((224,224)),
                     #t.RandomCrop((224,224)),
                     t.RandomHorizontalFlip(),
                     t.RandomVerticalFlip(),
                     #t.RandomAffine(degrees=(-180,180), translate=(0.1,0.1), scale=(0.9,1.1), shear=(-5,5)),
                     t.ToTensor()])
dset_train=torchvision.datasets.ImageFolder(root=train_path,transform=transform)

test_trans=t.Compose([t.Resize((224,224)),t.ToTensor()])
#dset_test=torchvision.datasets.ImageFolder(root=test_path,transform=test_trans)
dset_val=torchvision.datasets.ImageFolder(root=val_path,transform=test_trans)

train_loader=torch.utils.data.DataLoader(dset_train,batch_size=batch_s,shuffle=True,num_workers=16)#,drop_last=True)
val_loader=torch.utils.data.DataLoader(dset_val,batch_size=batch_s,shuffle=False,num_workers=16)#,drop_last=True)
#test_loader=torch.utils.data.DataLoader(dset_test,batch_size=batch_s,num_workers=16)#, drop_last=True)

num_classes = 2
#net=model.googlenet()

############################## MODEL ########################################




models = torchvision.models.googlenet(pretrained=True)




class MyModel(nn.Module):
  def __init__(self):
    super(MyModel, self).__init__()
    img_modules = list(models.children())[:-1]
    self.ModelA = nn.Sequential(*img_modules)
    self.relu = nn.ReLU()
    self.Linear3 = nn.Linear(1024, num_classes, bias = True)

  def forward(self, x):
    x = self.ModelA(x) # N x 1024 x 1 x 1
    x1 = torch.flatten(x, 1) 
    x2 = self.Linear3(x1)

    return  x1, x2




net = MyModel()
net=net.cuda()
criterion=nn.CrossEntropyLoss()
params = net.parameters()
optimizer=torch.optim.Adam(net.parameters())
model_name1 = model_name

load_model=snapshot_path+'/model_'+model_name+'.pth'
loaded_flag=False
if os.path.exists(load_model):
    checkpoint=torch.load(load_model)
    net.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print("model loaded successfully")
    print('starting training after epoch: ',checkpoint['epoch'])
    loaded_flag=True
    

def plot(val_loss,train_loss):
    plt.title("Loss after epoch: {}".format(len(train_loss)))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(list(range(len(train_loss))),train_loss,color="r",label="Train_loss")
    plt.plot(list(range(len(val_loss))),val_loss,color="b",label="Validation_loss")
    plt.legend()
    plt.savefig(os.path.join(plot_path,"loss_"+model_name+".png"))
    #plt.figure()
    plt.close()



val_interval=1
max_acc = 0.0
min_loss=99999
val_loss_gph=[]
train_loss_gph=[]


if loaded_flag:
    min_loss=checkpoint['loss']
    val_loss_gph=checkpoint["val_graph"]
    train_loss_gph=checkpoint["train_graph"]


########################## TRIAN ##################

def train(epoch=5):
  i=0
  global min_loss
  flag=True
  while i+1<=epoch and flag:
    print("Epoch {}".format(i+1 if not loaded_flag else i+1+checkpoint['epoch']))
    train_loss=0.0
    i+=1
    data1 = []
    correct=total=0
    #net = net.train()
    for (image,label) in train_loader:
      net.train()
      optimizer.zero_grad()
      outputs1, outputs2=net(image.cuda())
      #data1.append(outputs1)
      loss=criterion(outputs2 ,label.cuda())
      loss.backward()
      optimizer.step()
      train_loss+=loss.item()*image.size(0)
      _, predicted = torch.max(outputs2.data, 1)
      total += label.size(0)
      correct += (predicted == label.cuda()).sum().item()
    print("Train accuracy", (100*correct/total))
    train_loss_gph.append(train_loss/len(dset_train))
    #net = net.eval()
    
    if (i+1)%val_interval==0 or (i+1)==epoch:
        net.eval()
        with torch.no_grad():
          val_loss=0
          correct=total=0
          for (img_v,lab_v ) in val_loader:
            output_v1, output_v2=net(img_v.cuda())
            #data1.append(output_v1)
            #val_loss+=criterion(output_v2,lab_v.cuda())
            val_loss+=criterion(output_v2,lab_v.cuda())*img_v.size(0)
            _, predicted = torch.max(output_v2.data, 1)
            total += lab_v.size(0)
            correct += (predicted == lab_v.cuda()).sum().item()
          print("Val accuracy", (100*correct/total))
          val_acc = 100*correct/total
          val_loss_gph.append(val_loss/len(dset_val))
        
          if min_loss>val_loss:
            state={
                "epoch":i if not loaded_flag else i+checkpoint['epoch'],
                "model_state":net.cpu().state_dict(),
                "optimizer_state":optimizer.state_dict(),
                "loss":min_loss,
                "train_graph":train_loss_gph,
                "val_graph":val_loss_gph,
            }
            
            min_loss=val_loss
            torch.save(state,os.path.join(snapshot_path,"model_"+model_name+'.pt'))
            net.cuda()
          print("validation loss : {:.6f} ".format(val_loss/len(dset_val)))
    plot(val_loss_gph,train_loss_gph)
    print("Train loss : {:.6f}".format(train_loss/len(dset_train)))
    if i==epoch:
      flag=False
      break
  
train(100)


#print("validation MIN loss obtained: {:.6f}".format(min_loss))
net=net.eval()
correct = 0
total = 0
data1 = []
with torch.no_grad():
      for data in train_loader:
          images, labels = data
          labels=labels.cuda()
          outputs1, outputs2 = net(images.cuda())
          data1.append(outputs1)
          _, predicted = torch.max(outputs2.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

print('Accuracy of the network on the train images: %d %%' % (
      100 * correct / total))

#print("validation MIN loss obtained: {:.6f}".format(min_loss))
net=net.eval()
correct = 0
total = 0
data2 = []
with torch.no_grad():
      for data in val_loader:
          images, labels = data
          labels=labels.cuda()
          outputs1, outputs2 = net(images.cuda())
          data2.append(outputs1)
          _, predicted = torch.max(outputs2.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

print('Accuracy of the network on the validation images: %d %%' % (
      100 * correct / total))


from google.colab import drive
drive.mount('/content/drive')

############### LOADING THE MODEL SAVED AT LAST EPOCH ###########3

load_model=snapshot_path+'/model_'+model_name+'.pt'
loaded_flag=False
if os.path.exists(load_model):
    checkpoint=torch.load(load_model)
    net.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print("model loaded successfully")
    print('starting training after epoch: ',checkpoint['epoch'])
    loaded_flag=True

###### LOADING DATA WITH BATCH SIZE 1 ################

#test_trans=t.Compose([t.Resize((224,224)),t.ToTensor()])
dset_train=torchvision.datasets.ImageFolder(root=train_path,transform=test_trans)


dset_val=torchvision.datasets.ImageFolder(root=val_path,transform=test_trans)


train_loader=torch.utils.data.DataLoader(dset_train,batch_size=1,shuffle=False,num_workers=16)#,drop_last=True)
val_loader=torch.utils.data.DataLoader(dset_val,batch_size=1,shuffle=False,num_workers=16)#,drop_last=True)


############### EXTRACTION OF FEATURES ############

net = net.cuda()

print("Train MIN loss obtained: {:.6f}".format(min_loss))
net=net.eval()
correct = 0
total = 0
data1 = []
train_label = []
with torch.no_grad():
      for data in train_loader:
          images, labels = data
          labels=labels.cuda()
          outputs1, outputs2 = net(images.cuda())
          data1.append(outputs1)
          train_label.append(labels)
          _, predicted = torch.max(outputs2.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

print('Accuracy of the network on the train images: %d %%' % (
      100 * correct / total))

print("validation MIN loss obtained: {:.6f}".format(min_loss))
net=net.eval()
correct = 0
total = 0
data2 = []
val_label = []
with torch.no_grad():
      for data in val_loader:
          images, labels = data
          labels=labels.cuda()
          outputs1, outputs2 = net(images.cuda())
          data2.append(outputs1)
          val_label.append(labels)
          _, predicted = torch.max(outputs2.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
      100 * correct / total))


data_all = data1+data2
total_label = train_label+val_label

####### LOADING THE CSV #############

#temp=data_all
temp = data_all
import csv
labels=[]
for i in range(len(temp[0].tolist()[0])):
  labels.append("GoogLeNet"+str(i+1))
with open ("/content/drive/MyDrive/Sars-cov-2/GoogLeNet_68%.csv",'w+',newline='') as file:
  writer=csv.writer(file)
  writer.writerow(labels)
  for i in range(len(temp)):
    row=temp[i].tolist()[0]
    writer.writerow(row)

### LOADING THE LABELS ### 

import pandas as pd

labels = []
for label in total_label:
    labels.append(label.tolist()[0])

df = pd.DataFrame(labels)
df.to_csv('/content/drive/MyDrive/Sars-cov-2/labels.csv')
