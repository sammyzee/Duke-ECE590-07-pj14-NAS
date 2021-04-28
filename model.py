
# coding: utf-8

# In[ ]:


import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from operation import *


# In[ ]:


'''
matrix =
         [[0 1 1 1]
          [0 0 1 0]
          [0 0 0 1]
          [0 0 0 0]]
ops = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'output']

'''
#The structure of the cell

class cell(nn.Module):
    def __init__(self, c_in, c_out):
        super(cell, self).__init__()
        self.input_out_1x1 = nn.Conv2d(c_in, c_in, kernel_size=(1,1),stride=1,padding=0)
        
        self.node1_1x1 = nn.Conv2d(c_in, c_out, kernel_size=(1,1),stride=1,padding=0)
        self.node1 = conv1x1(c_out, c_out)
        
        self.node2_1x1 = nn.Conv2d(c_in, c_out, kernel_size=(1,1),stride=1,padding=0)
        self.node2 = conv3x3(c_out, c_out)
    
    def forward(self, x):
        residual = self.input_out_1x1(x)
        
        out1 = self.node1(self.node1_1x1(x))
        out2 = self.node2(out1+self.node2_1x1(x))
        
        return residual + out2

    

    
class stack(nn.Module):
    def __init__(self, c_in, c_out, num):
        super(stack, self).__init__()
        
        self.cellstack = nn.ModuleList()
        for i in range(num):
            layer = cell(c_in, c_out)
            self.cellstack.append(layer)
        
    def forward(self, x):
        out = x
        for idx, cell in enumerate(self.cellstack):
            out = cell(out)
            
        return out



class model_nas(nn.Module):
    def __init__(self, c_in, init_chs, num_cell, num_stack, classes):
        super(model_nas, self).__init__()
        
        self.stemconv = nn.Conv2d(c_in, init_chs, kernel_size=(3,3), stride=1, padding=(1,1))
        
        self.model = nn.ModuleList()
        self.chs = init_chs
        for i in range(num_stack):
            cellstack = stack(self.chs, self.chs, num_cell)
            self.model.append(cellstack)
            
            if i==num_stack-1:
                sampling = global_avg(self.chs, 2*self.chs)
            else:
                sampling = downsampling(self.chs, 2*self.chs)
            self.model.append(sampling)
            
            self.chs = 2*self.chs
        
        self.FC = nn.Linear(self.chs, classes)
        
    
    def forward(self, x):
        out = self.stemconv(x)
            
        for idx, stack in enumerate(self.model):
            out = stack(out)
                
        out = self.FC(out)
            
        return out

