import torch
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import scipy.io as sio
import numpy as np
import torchvision.models as models
from sklearn import cluster
from munkres import Munkres
from PIL import Image
from model import *
from utill import Custom_dataset
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
import argparse
#from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from tqdm import tqdm
import itertools


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='PyTorch DSC vanish')

parser.add_argument('--batch' ,type=int, default=1, metavar='N')
parser.add_argument('--num_class', type=int, default=4,metavar='N')
parser.add_argument('--epoch', type=int, default=1000,metavar='N')
learning_rate = 1e-3
alpha = 0.04


args = parser.parse_args()




####### dataset ########
transform = transforms.Compose([ transforms.ToTensor(),])



data_path = ""

train_dataset = Custom_dataset(transform=transform,path=data_path)

trainloader = DataLoader(train_dataset,batch_size=args.batch,shuffle=None)

####### model ########

pretrain_model_PATH = './model_weight/model_weight.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Network(True).to(device)


model.load_state_dict(torch.load(pretrain_model_PATH),strict=False)

optimizer = optim.Adam((model.parameters()), lr=learning_rate)

crition=nn.MSELoss(reduction='sum')

####### test function ########
def best_map(L1, L2):
    # L1 should be the labels and L2 should be the clustering number we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def thrC(C, ro):
    if ro < 1:

        N = C.shape[0]

        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L


def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate


lambda_1 = 100
lambda_2 = 1



best_acc=0
best_epoch=0
is_best=0

####### train ########

for epoch in tqdm(range(args.epoch)):

    for i, data in enumerate(trainloader, 0):
        model.train()

        inputs, _ = data
        inputs = inputs.to(device)


        optimizer.zero_grad()

        output , z_conv, z_ssc, Coef = model(inputs)
        
        recon_loss = crition(inputs,output)*0.5
        reg_loss = lambda_1 * torch.sum(torch.pow(Coef,2))
        ssc_loss = lambda_2 * crition(z_ssc,z_conv)
        


        loss = recon_loss + ssc_loss + reg_loss
        loss.backward()
        
        optimizer.step()



        


    
    with torch.no_grad():
        model.eval()

        for data in trainloader:
        images, labels = data
        images=images.to(device)
        labels = labels.numpy()
        
        _ , _, _, Coef = model(inputs)

        Coef = Coef.cpu().numpy()


        C = thrC(Coef, alpha)
        
        y_x, CKSym_X = post_proC(C, 4, 12, 8)
        
        missrate_x = err_rate(labels, y_x)
        
        acc = 1 - missrate_x
        if acc > best_acc:
            best_acc=acc
            torch.save(model.state_dict(),'./model_weight/model_weight_best.pth')


f = open("./acc.txt",'a')

f.write(best_acc "\n")

f.close()
