#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch import nn
from torch.optim import Adam
from .basic_module import BasicModule
from torchvision.models import resnet18


# In[6]:


class ResNet(BasicModule):
    def __init__(self,out_features=1):
        super(ResNet,self).__init__()
        self.model_name = 'resnet'
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(in_features=512,out_features=out_features,bias=True)
        
    def forward(self,x):
        return self.model(x)
    