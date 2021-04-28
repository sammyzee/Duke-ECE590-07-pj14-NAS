
# coding: utf-8

# In[ ]:


import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


# Basic Operation: con1x1-bn-re, con3x3-bn-re
class conv1x1(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=(1,1), stride=1, padding=0):
        super(conv1x1, self).__init__()
        self.op =                 nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding),                                 nn.BatchNorm2d(c_out),                                 nn.ReLU())
        
    def forward(self, x):
        return self.op(x)
    

    
class conv3x3(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=(3,3), stride=1, padding=(1, 1)):
        super(conv3x3, self).__init__()
        self.op =                 nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding),                               nn.BatchNorm2d(c_out),                               nn.ReLU())
        
    def forward(self, x):
        return self.op(x)
    
    
    
class downsampling(nn.Module):
    def __init__(self, c_in, c_out):
        super(downsampling, self).__init__()
        self.op =                 nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=1, padding=0),                               nn.MaxPool2d(kernel_size=(1,1), stride=2, padding=0))
        
    def forward(self, x):
        return self.op(x)
    

    
class global_avg(nn.Module):
    def __init__(self, c_in, c_out):
        super(global_avg, self).__init__()
        self.op =                 nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=(1, 1), stride=1, padding=0))
        
    def forward(self, x):
        out = self.op(x)
        out = out.mean(dim=(-2, -1))
        
        return out

