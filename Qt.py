#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog
from PyQt5.Qt import QPixmap, QPoint, QPainter
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from test import test


# In[2]:


class ImageBox(QWidget):
    def __init__(self):
        super(ImageBox,self).__init__()
        self.img = None
        self.scaled_img = None
        self.point = QPoint(0,0)
        self.start_pos = None
        self.end_pos = None
        self.left_click = False
        self.scale = 1
    
    def init_ui(self):
        self.setWindowTitle('ImageBox')
        
    def set_image(self,image_path):
        self.img = QPixmap(image_path)
        if self.img.width()>640 or self.img.height()>640:
            self.scale = 0.5
        
    def paintEvent(self,e):
        if self.img:
            painter = QPainter()
            painter.begin(self)
            painter.scale(self.scale,self.scale)
            painter.drawPixmap(self.point,self.img)
            painter.end()
            
    def mouseMoveEvent(self,e):
        if self.left_click:
            self.end_pos = e.pos() - self.start_pos
            self.point = self.point + self.end_pos
            self.start_pos = e.pos()
            self.repaint()
            
    def mousePressEvent(self,e):
        if e.button() == Qt.LeftButton:
            self.left_click = True
            self.start_pos = e.pos()
            
    def mouseReleaseEvent(self,e):
        if e.button() == Qt.LeftButton:
            self.left_click = False


# In[3]:


class MainDemo(QWidget):
    def __init__(self):
        super(MainDemo,self).__init__()
        
        self.setWindowTitle('Image Viewer')
                
        self.open_file = QPushButton('Open Image')
        self.open_file.clicked.connect(self.open_image)
        self.open_file.setFixedSize(150,30)
        
        self.beauty_file = QPushButton('Prediction')
        self.beauty_file.clicked.connect(self.start_pred)
        self.beauty_file.setFixedSize(150,30)
        
        w = QWidget(self)
        layout = QHBoxLayout()
        layout.addWidget(self.open_file)
        layout.addWidget(self.beauty_file)
        layout.setAlignment(Qt.AlignLeft)
        w.setLayout(layout)
        w.setFixedSize(550,50)
        
        c = QWidget(self)
        self.box = ImageBox()
        self.target = ImageBox()
        layout = QHBoxLayout()
        layout.addWidget(self.box)
        layout.addWidget(self.target)
        c.setLayout(layout)
        
        layout = QVBoxLayout()
        layout.addWidget(w)
        layout.addWidget(c)
        self.setLayout(layout)
        
    def open_image(self):
        img_name,_ = QFileDialog.getOpenFileName(self,"Open Image File", "*.jpg;;*.png;;*.jpeg")
        self.box.set_image(img_name)
        self.predBeauty_thread = PredBeauty(img_name)
        self.predBeauty_thread.status_signal.connect(self.pred_image)
    
    def start_pred(self):
        if hasattr(self,'predBeauty_thread'):
            self.open_file.setEnabled(False)
            self.beauty_file.setEnabled(False)
            self.predBeauty_thread.start()
    
    def pred_image(self,result_path):
        self.target.set_image(result_path)
        os.remove(result_path)
        self.open_file.setEnabled(True)
        self.beauty_file.setEnabled(True)


# In[4]:


class PredBeauty(QThread):
    status_signal = pyqtSignal(str)
    
    def __init__(self,img_name):
        super(PredBeauty,self).__init__()
        self.img_name = img_name
    
    def run(self):
        result_path = test(test_file=self.img_name)
        self.status_signal.emit(result_path)


# In[5]:

if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = MainDemo()
    demo.show()
    sys.exit(app.exec_())

