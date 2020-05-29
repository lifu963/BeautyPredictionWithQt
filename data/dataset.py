#!/usr/bin/env python
# coding: utf-8

# In[123]:


import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import pandas as pd


# In[124]:


class BeautyDataset(data.Dataset):
    def __init__(self,root,label_path,transforms=None,train=True):
        self.train = train
        
        imgs = [os.path.join(root,img) for img in os.listdir(root)]
        imgs = sorted(imgs,key= lambda x:(x.split('/')[-1].split('.')[0]))
        
        imgs_num = len(imgs)
        if self.train:
            self.imgs = imgs[:int(0.8*imgs_num)]
        else:
            self.imgs = imgs[int(0.8*imgs_num):]
        
        self.labels = pd.read_csv(label_path)
        
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            
            if self.train:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
    
    def __getitem__(self,index):
        img_path = self.imgs[index]
        
        filename = img_path.split('/')[-1]
        label = self.labels[self.labels.Filename==filename].score.item()
        
        data = Image.open(img_path)
        data = self.transforms(data)
        return data,label
    
    def __len__(self):
        return len(self.imgs)

