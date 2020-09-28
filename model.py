import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

def init_weight(m):
    if type(m)==(nn.Conv2d or nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
#########Coil 20############
class Conv_layer(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride,padding):
        super(Conv_layer, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
       
        self.relu = nn.ReLU(True) 


    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
    
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_channel, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, reduction_channel),
            nn.ReLU(),
            nn.Linear(reduction_channel, gate_channels)
            )
        self.pool_types = pool_types
        self.global_avg_pool=nn.AdaptiveAvgPool2d(1)
        self.global_max_pool=nn.AdaptiveMaxPool2d(1)
    def forward(self, x):
        channel_att_sum = None
        
        for pool_type in self.pool_types:
            if pool_type=='avg':
                
                avg_pool = self.global_avg_pool(x)
               
                avg_pool = avg_pool.view(avg_pool.size(0), -1)
                
  #              avg_pool = torch.unsqueeze(avg_pool, 2) 
                
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = self.global_max_pool(x)
                max_pool = max_pool.view(avg_pool.size(0), -1)
  #              max_pool = torch.unsqueeze(max_pool, 2) 
                channel_att_raw = self.mlp( max_pool )
            

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = nn.Sigmoid()( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale



class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self,kernel_size,padding):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2,1,3,stride=1,padding=1)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = nn.Sigmoid()(x_out) 
        
        return x * scale



class Network(nn.Module):
    def __init__(self,test=False):
        super(Network, self).__init__()

        ####### Encoder ########
        
        self.encoder_1 = nn.Conv2d(in_channels = 1,out_channels = 10, kernel_size = 5,stride=2, padding = 5//2)
        self.encoder_2 = nn.Conv2d(in_channels = 10,out_channels = 20,kernel_size = 3,stride=2, padding = 3//2)
        self.encoder_3 = nn.Conv2d(in_channels = 20,out_channels = 30,kernel_size = 3,stride=2, padding = 3//2)

        ####### Decoder ########
        
        self.decoder_1 =nn.ConvTranspose2d(in_channels = 30, out_channels = 20, kernel_size = 3, stride=2, padding = 3//2)
        self.decoder_2 =nn.ConvTranspose2d(in_channels = 20, out_channels = 10, kernel_size = 3, stride=2, padding = 3//2)
        self.decoder_3 =nn.ConvTranspose2d(in_channels = 10, out_channels = 1, kernel_size = 5, stride=2, padding = 5//2)

        ####### Self-Expressive layer ########

        self.Coef = nn.Parameter(1.0e-4 * torch.ones((4, 4)))

        ####### Utill ########
        self.Relu= nn.ReLU()
        self.test=test
        self.shape=[]

        

    def forward(self, x):
       self.shape.append(x.shape)
       
       if self.test == True:

            ####### Encoder 1 ########
            x = self.encoder_1(x)
            x = self.Relu(x)
            self.shape.append(x.shape)

            ####### Encoder 2 ########
            x = self.encoder_2(x)
            x = self.Relu(x)
            self.shape.append(x.shape)

            ####### Encoder 3 ########
            x = self.encoder_3(x)
            x = self.Relu(x)
            self.shape.append(x.shape)

            ####### Self-Expressive layer ########
            z_conv,z_ssc = self.self_expressive(x)
            x = torch.reshape(z_ssc,self.shape[3])

            ####### Decoder 1 ########   
            x = self.decoder_1(x,output_size=self.shape[2])
            x = self.Relu(x)

            ####### Decoder 2 ########
            x = self.decoder_2(x,output_size=self.shape[1])
            x = self.Relu(x)

            ####### Decoder 3 ########
            x = self.decoder_3(x,output_size=self.shape[0])
            x = self.Relu(x)

            return x ,z_conv,z_ssc,self.Coef

       else: 

            ####### Encoder 1 ########
            x = self.encoder_1(x)
            x = self.Relu(x)
            self.shape.append(x.shape)

            ####### Encoder 2 ########
            x = self.encoder_2(x)
            x = self.Relu(x)
            self.shape.append(x.shape)

            ####### Encoder 3 ########
            x = self.encoder_3(x)
            x = self.Relu(x)
            self.shape.append(x.shape)


            ####### Decoder 1 ########   
            x = self.decoder_1(x,output_size=self.shape[2])
            x = self.Relu(x)

            ####### Decoder 2 ########
            x = self.decoder_2(x,output_size=self.shape[1])
            x = self.Relu(x)

            ####### Decoder 3 ########
            x = self.decoder_3(x,output_size=self.shape[0])
            x = self.Relu(x)

        
            return  x

    def self_expressive(self,x):
        z_conv = x.view(4,-1)
        z_ssc = torch.matmul(self.Coef, z_conv)
        return z_conv,z_ssc  




