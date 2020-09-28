import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from model import *
from utill import Custom_dataset
from torchvision import datasets, transforms
import argparse
from torch.utils.data import Dataset
import numpy as np
import torch.nn.init as init
from tqdm import tqdm
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
parser = argparse.ArgumentParser(description='PreTrain DSC')


parser.add_argument('--batch' ,type=int, default=8, metavar='N')
parser.add_argument('--epoch', type=int, default=1200,metavar='N')



args = parser.parse_args()


####### dataset ########
transform = transforms.Compose([transforms.ToTensor(), ])

data_path = 

train_dataset = Custom_dataset(transform=transform,path=data_path)

trainloader=DataLoader(train_dataset,batch_size=args.batch,shuffle=True)

####### model ########
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Network().to(device)

model.apply(init_weight)


optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion=nn.MSELoss(reduction="sum")

####### train ########

for epoch in tqdm(range(args.epoch)):
    model.train()

    running_loss = 0.0


    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, _ = data

        inputs = inputs.to(device)
        
        optimizer.zero_grad()
        outputs= model(inputs)
        
        loss = criterion(outputs, inputs)


        loss.backward()
        optimizer.step()
        running_loss += loss.item()
       
#    print("epoch : {} , loss: {:.8f}".format(epoch + 1, running_loss / len(trainloader)))
    

torch.save(model.state_dict(),'./model_weight/model_weight.pth')
