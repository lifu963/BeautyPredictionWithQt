#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import dlib
import torch
from config import opt
import models
from PIL import Image


# In[2]:
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
            cv2.putText(image, 'Value:' + str(res), (lefttop_x - 5, lefttop_y - 5), font, 1, (0, 0, 255), 2)
    
    result_path = './result.jpg'
    cv2.imwrite(result_path,image)
    return result_path
#     测试用
#     cv2.namedWindow('result', cv2.WINDOW_AUTOSIZE)
#     cv2.imshow('result', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



if __name__ == '__main__':
    import fire
    fire.Fire()