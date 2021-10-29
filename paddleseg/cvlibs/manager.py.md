## manager.py 代码解析
* 设计思想
1. 使用了面向对象的设计思想，manage有5个ComponentManager对象，
   他们分别是MODELS、BACKBONES、DATASETS、TRANSFORMS和LOSSES。这5个ComponentManager类似字典，
   用来维护套件中所有对应的类，比如FCN类、ResNet类等，通过类的名称就可以找到对应的类  
2. 


## 参考链接
* 1 
