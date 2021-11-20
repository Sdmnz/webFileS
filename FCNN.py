import numpy as np
import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import from_numpy, tensor
from torch.utils.data import Dataset
import time
        
    
class My_Data(Dataset):
    """ Diabetes dataset."""
    # Initialize your data, download, etc.
    def __init__(self,X,y):
        self.len = X.shape[0]
        self.d_dim = X.shape[1]
        self.d_out = np.unique(y).shape[0]
        self.x_data = from_numpy(X).float()
        self.y_data = from_numpy(y).int()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len
    
class My_Data2(Dataset):
    """ Diabetes dataset."""
    # Initialize your data, download, etc.
    def __init__(self,X):
        self.len = X.shape[0]
        self.d_dim = X.shape[1]
        self.x_data = from_numpy(X).float()

    def __getitem__(self, index):
        return self.x_data[index]
    
    def __len__(self):
        return self.len
    
class NeuralNetwork(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NeuralNetwork,self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,num_classes)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
    
class FCNN():
    def __init__(self,X_train_FE,y_train,**kwards):
        batch_size = kwards.get('batch_size',1)
        self.device = kwards.get('device',torch.device('cpu'))  
        self.learning_rate = kwards.get('learning_rate', 0.001)
        self.batch_size = batch_size
        

        # # Initialize Network 
        self.num_classes = np.unique(y_train).shape[0]
        self.input_size =  X_train_FE.shape[1]
        try:
            self.model = NeuralNetwork(input_size = self.input_size,
                        num_classes=self.num_classes).to(self.device)
        except:
            self.device = 'cpu'
            self.model = NeuralNetwork(input_size = self.input_size,
                        num_classes=self.num_classes).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
                
        self.train_dataset = My_Data(X_train_FE,y_train)
        
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True)
        
    def fit(self,**kwards):
        num_epochs = kwards.get('num_epochs',20)
        
        # Training Network 
        for epoch in range(num_epochs):
            # Get data to cuda if possible
            for batch_idx, (data,targets) in enumerate(self.train_loader):    
                data = data.to(device = self.device)
                targets = targets.to(device=self.device,dtype=torch.int64)
                
                #get to correct shape
                data = data.reshape(data.shape[0],-1)
                
                # forward
                scores = self.model(data)
                loss =  self.criterion(scores,targets)
                
                #backward
                self.optimizer.zero_grad()
                loss.backward()
                
                # gradient descent or adam step
                self.optimizer.step()
            print(f'Epoch {epoch + 1} | Batch: {batch_idx+1} | Loss: {loss.item():.10f}')
            
        self.check_accuracy()
        
        
    def predict(self,X_FE):
        data_Row = My_Data2(X_FE)
        Loader_Row = DataLoader(dataset = data_Row,
                                batch_size=1)
        
        out = self.Chek_Model(Loader_Row)
        return out
        
        
    def Chek_Model(self,loader):
        # if loader.dataset.train:
        #     print('accurancy of training data')
        # else:
        #     print('Checking accurancy on test data')
        self.model.eval()  
        Lab_out = []
        with torch.no_grad():
            for x in loader:
                x = x.to(device = self.device)
                x = x.reshape(x.shape[0],-1)
                scores = self.model(x)
                _,predictions = scores.max(1)             
                Lab_out.append(predictions)         
        self.model.train()
        return np.array(Lab_out)
    
    
    def check_accuracy(self):
        loader = self.train_loader
        num_corrrect = 0
        num_samples = 0
        self.model.eval()    
        Lab_out = []
        with torch.no_grad():
            for x,y in loader:
                x = x.to(device = self.device)
                y = y.to(device = self.device,dtype=torch.int64)
                x = x.reshape(x.shape[0],-1)
                scores = self.model(x)
                _,predictions = scores.max(1)             
                Lab_out.append(predictions)
                num_corrrect += (predictions == y).sum()
                num_samples += predictions.size(0)
                
            print(f'Got {num_corrrect}/{num_samples} with acurancy \
                  {float(num_corrrect)/float(num_samples)*100:.2f}')
        self.model.train()
        return Lab_out
            


def FCNN_HSI(HSI):
    Parameters ={
        'batch_size':1,
        'num_epochs':20,
        'device':'cuda'
        }
    X_train = HSI.getX_train()
    y_train = HSI.gety_train()

    Model = FCNN(X_train,y_train,**Parameters)        
    Model.fit(**Parameters)
    
    y_Raw_pred = Model.predict(HSI.X_Raw) 
    for i,j in enumerate(y_Raw_pred):
        y_Raw_pred[i] = j.numpy()[0]
    
    y_Raw_pred = y_Raw_pred.astype('float64')
    return y_Raw_pred

        
        
    
    



    
    
