## CRNN的mobilenet实现 旨在构建轻量级文字识别网络

网络的实现参考了 https://github.com/AstarLight/Lets_OCR

cnn部分采用的是mobilenetv3的bneckbone
rnn部分的hidden_unit从原来的256缩减为48（参考百度平台的[ppocr](https://github.com/PaddlePaddle/PaddleOCR)）



todo：

* 在原先的基础上进行数据增广的操作


数据增广方法使用的是 [TIA 的python版本实现](https://github.com/RubanSeven/Text-Image-Augmentation-python)，该方法可以将图像进行更加灵活的变形。