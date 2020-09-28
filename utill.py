from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio


class Custom_dataset(Dataset):
    def __init__(self, transform, path):
        # Transforms
        self.transfrom = transform


       
        

    def __getitem__(self, idx):
        label = [0,1,2,3]

        return 

    def __len__(self):

        
        return 