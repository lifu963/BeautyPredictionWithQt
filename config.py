#!/usr/bin/env python
# coding: utf-8

# In[1]:

from torchvision import transforms as T
import warnings


# In[2]:
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            
transforms = T.Compose([    
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])


class DefaultConfig(object):
    env = 'default'
    vis_port = 8097
    gpu = False
    
    root = '../SCUT-FBP5500_v2/Images/'
    label_path = 'label.csv'
    
    model = 'ResNet'
    load_model_path = 'checkpoints/resnet_27_86.pth'
    
    batch_size = 64
#     use_gpu = True
    num_workers = 4
    print_freq = 20
    
    test_file = 'AF2.jpg'
    
    max_epoch = 50
    lr = 0.001
    lr_decay = 0.5
    weight_decay = 0e-5
    error_tolerance = 0.5
    default_transform = transforms
    
    def _parse(self,kwargs):
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn('Warning: opt has not attribut %s'%k)
            setattr(self,k,v)
        
        print('use config:')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k,getattr(self,k))


# In[3]:


opt = DefaultConfig()

