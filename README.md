# BeautyPredictionWithQt
封装为小程序的颜值预测器。

数据集地址：https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
获取数据集后，我将评分数据处理为csv格式，方便读取。但那时是在jupyter notebook上完成数据的处理，处理后就删除了该文件。
因此这部分源码缺失(hhh)，因此我就将处理后的label.csv直接上传上来了。

data/内的代码，将数据集基于data.Dataset处理为新类，方便读取。
model/内的代码主要是构建模型。
本项目使用了Resnet18模型，并重新训练。
utils/内提供了训练时可视化的工具。
main.py 中的train函数完成了对模型的训练。
test.py 中的test函数可以生成预测结果，并作为Qt.py的接口。
在Qt.py 中完成了将脚本构建成小程序的工作。
训练前请创建checkoutpoints文件夹存放训练完后生成的模型。
