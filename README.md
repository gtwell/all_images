### <font color=RoyalBlue size=5 face="黑体">
## This is my new program that classifies cashmere and wool with convolutional neural network(CNN). This work is finished with PyTorch.
* ### [1.读取数据](#1)
* ### [2.选择模型](#2)
    * #### [2.1 ResNet](#2.1)
    * #### [2.2 Inception](#2.2)
* ### [3.优化、训练](#3)
    * #### [3.1 optimizer](#3.1)
    * #### [3.2 learning_rate](#3.2)
* ### [4.评估](#4)
</font>

---
### <span id="1">1.读取数据</span>
新建一个读取数据的dataset文件，在该文件中写入一个MyImagesDataset类进行数据读取和处理。
