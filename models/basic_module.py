#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import time


# In[2]:


class BasicModule(torch.nn.Module):
    
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(type(self))
        
    def load(self,path):
        self.load_state_dict(torch.load(path))
        
    def save(self,name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix+'%m%d_%H_%M_%S.pth')
        torch.save(self.state_dict(),name)
        return name
    
    def get_optimizer(self,lr,weight_decay):
        return            torch.optim.Adam(filter(lambda p:p.requires_grad,self.parameters()),lr=lr,weight_decay=weight_decay)


# In[3]:


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()
        
    def forward(self,x):
        return x.view(x.size(0),-1)

