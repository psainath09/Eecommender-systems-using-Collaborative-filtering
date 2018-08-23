# basic code has been taken from super "https://www.superdatascience.com/deep-learning/"
# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt

# research data import
data= pd.read_csv('recommender_data.csv')
data= data.drop(["user_sequence"],axis=1)
data["course"]= data["course_id"].apply(lambda x: x[2:])
data= data.drop(["course_id"],axis=1)
data["course"]= pd.to_numeric(data["course"])
data["completed"]=1
data=data.assign(id=(data["course"]).astype('category').cat.codes)
#split data into train and test
from sklearn.cross_validation import train_test_split
train_set,test_set=train_test_split(data,test_size=0.25)
# count of users and courses
users=len(data["learner_id"].unique())
courses=len(data["course"].unique())
train_set=np.array(train_set)
test_set= np.array(test_set)
# Converting the data into an array with learners in lines and courses in columns
def convert(datas):
    new_data = []
    for id_users in range(1, users + 1):
        id_courses = datas[:,2][datas[:,0] == id_users]
        id_done = datas[:,3][datas[:,0] == id_users]
        done = np.zeros(courses)
        done[datas[:,4][np.where(id_courses)]] = id_done
        new_data.append(list(done))
    return new_data

train_set = convert(train_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
train_set = torch.FloatTensor(train_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
# parameters for RBM model
nv = len(train_set[0])  
nh = 100
batch_size = 20000
rbm = RBM(nv, nh)

# Training the RBM and calculate MAE(mean absolute error)
epoch_= []
loss_=[]
nb_epoch = 6
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, users - batch_size, batch_size):
        vk = train_set[id_user:id_user+batch_size]
        v0 = train_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    epoch_.append(epoch)
    loss_.append(train_loss/s)
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM using MAE(mean aboslute error)
loss_test_abs=[]
test_loss = 0
s = 0.
for id_user in range(users):
    v = train_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
loss_test_abs.append(test_loss/s)
print('test loss: '+str(test_loss/s))

# Training the RBM and calculate RMSE
epoch_rmse= []
loss_rmse=[]
nb_epoch = 6
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, users - batch_size, batch_size):
        vk = train_set[id_user:id_user+batch_size]
        v0 = train_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2))
        s += 1.
    epoch_rmse.append(epoch)
    loss_rmse.append(train_loss/s)
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM using RMSE

loss_test_rmse=[]
test_loss = 0
s = 0.
for id_user in range(users):
    v = train_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2))
        s += 1.
loss_test_rmse.append(test_loss/s)
print('test loss: '+str(test_loss/s))

train_epochs_results= pd.DataFrame({"Epoch":epoch_,"RMSE_loss_Error":loss_rmse,"AD":loss_})
test_loss_results= pd.DataFrame({"RMSE_test_Error":loss_test_rmse,"AD_test_error":loss_test_abs})

train_epochs_results= pd.read_csv("train_results.csv")
test_loss_results=pd.read_csv("test_result.csv")
test_loss_results.columns=["1","MAE","RMSE"]

#plt.figure(figsize=((16,4))) plot 
plt.plot(train_epochs_results["Epoch"], train_epochs_results["RMSE_loss_Error"], 
         color='black', marker='*', linestyle='dashed',
        linewidth=2, markersize=12,markeredgecolor="red")
plt.xlabel("Epochs")
plt.ylabel("Root Mean Square Error")
plt.grid(axis="y")
plt.savefig("rmse.png")

plt.plot(train_epochs_results["Epoch"], train_epochs_results["AD"], color='black', 
         marker='*', linestyle='dashed',
        linewidth=2, markersize=12,markeredgecolor="red")
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.grid(axis="y")
plt.savefig("mae.png")