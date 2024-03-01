
本项目基于项目：
Resnet50:https://github.com/dongtuoc/cv_learning_resnet50?tab=readme-ov-file


在win11下，通过VisualStudio2022搭建C++项目。

nNet：
实现了一个简单的神经网络模板，可以粗略的训练和直接加载使用训练好的模型。使用mnist训练集进行示例训练和预测，准确性和效率较低。
简单使用opencv2中的部分模块，例如：
1.template class Mat 矩阵形式存放神经网络中的数据，便于使用矩阵运算.
2.FileStorage 用于模型数据和训练集数据以XML文件的存取。

resnet50：
实现了resnet50的卷积网络模型。读取训练好后的resnet50模型，进行图片预测。
使用基础的template模板技术，但是仅支持浮点数float和double用于计算。
实现了ConvLayer,BatchNormLayer,ReluLayer,PoolLayer,FcLayer和Resnet提出的残差结构。



