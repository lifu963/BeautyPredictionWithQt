#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import warnings
import time
import torch
import torchvision
from torch import nn
from data.dataset import BeautyDataset
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm
import models
from config import opt
import cv2
import dlib
import numpy as np
from skimage.transform import resize
from torchvision import transforms as T
from PIL import Image

@torch.no_grad()
def test(**kwargs):
    opt._parse(kwargs)

    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    device = torch.device('cuda') if opt.gpu else torch.device('cpu')
    opt.device = device
    model.to(opt.device)

    detector = dlib.get_frontal_face_detector()
    image = cv2.imread(opt.test_file)
    b, g, r = cv2.split(image)
    image_rgb = cv2.merge([r, g, b])
    rects = detector(image_rgb, 1)
    if len(rects) >= 1:
        for rect in rects:
            lefttop_x = rect.left()
            lefttop_y = rect.top()
            rightbottom_x = rect.right()
            rightbottom_y = rect.bottom()
            cv2.rectangle(image, (lefttop_x, lefttop_y), (rightbottom_x, rightbottom_y), (0, 255, 0), 2)

            face = image_rgb[lefttop_y:rightbottom_y, lefttop_x:rightbottom_x]
            face_iamge = Image.fromarray(face)
            #           (c,h,w)
            transforms = opt.default_transform
            face = transforms(face_iamge).to(opt.device)
            #           (batch,c,h,w)
            face = face.unsqueeze(0)
            res = round(model(face).item(), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, 'Value:' + str(res), (lefttop_x - 5, lefttop_y - 5), font, 0.5, (0, 0, 255), 1)

    cv2.namedWindow('result', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def train(**kwargs): 
    opt._parse(kwargs)
    vis = Visualizer(opt.env,port = opt.vis_port)
    
    model = getattr(models,opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    device = torch.device('cuda') if opt.gpu else torch.device('cpu')
    opt.device = device
    model.to(opt.device)
    
    train_data = BeautyDataset(root=opt.root,label_path=opt.label_path,train=True)
    val_data = BeautyDataset(root=opt.root,label_path=opt.label_path,train=False)
    train_dataloader = DataLoader(train_data,opt.batch_size,shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    
    criterion = nn.MSELoss()
    optimizer = model.get_optimizer(lr=opt.lr,weight_decay=opt.weight_decay)
    
    loss_meter = meter.AverageValueMeter()
    previous_loss = 1e10
    
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        
        for i,(data,label) in tqdm(enumerate(train_dataloader)):
            input = data.to(opt.device)
            target = label.to(opt.device).type(torch.float32).reshape(-1,1)
            
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score,target)
            loss.backward()
            optimizer.step()
            
            loss_meter.add(loss.item())
            
            if (i+1)%opt.print_freq == 0:
                vis.plot('loss',loss_meter.value()[0])
                
        model.save()
        val_acc = val(model,val_dataloader)
        vis.plot('val_accuracy',val_acc)
        vis.plot('epoch:{epoch},lr:{lr},loss:{loss},val_acc:{val_acc}'.format(
        epoch=epoch,lr=opt.lr,loss=loss_meter.value()[0],val_acc=str(val_acc) ))
        
        if loss_meter.value()[0] > previous_loss:
            lr = lr*opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        previous_loss = loss_meter.value()[0]


# In[100]:


@torch.no_grad()
def val(model,dataloader):
    model.eval()
    n_correct = 0
    n_total = 0
    for i,(data,label) in tqdm(enumerate(dataloader)):
        input = data.to(opt.device)
        target = label.to(opt.device).type(torch.float32).reshape(-1,1)
        score = model(input)
        n_correct += (abs(target-score)<opt.error_tolerance).sum().item()
        n_total += data.size(0)
    acc = n_correct / n_total
    model.train()
    return acc


# In[110]:


def help():    
    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

# In[ ]:


if __name__ == '__main__':
    import fire
    fire.Fire()

