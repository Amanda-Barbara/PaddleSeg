# 图像分割基础教程
## 语义分割算法的根本目的：像素级分类！
![](./images/语义分割算法的基础概念.png)
![](./images/像素级分类.png)

* 全卷积从分类到分割
![](./images/全卷积分类到分割.png)
  

## 语义分割算法的基本流程
![](./images/语义分割算法的基本流程.png)  
  
## 语义分割性能指标
![](./images/语义分割性能指标.png)
![](./images/语义分割性能指标mAcc.png)
![](./images/语义分割性能指标mIoU.png)

### 图像分割类型
1. 语义分割：给每个pixel分类
2. 实例分割：给每个框里的object分mask
3. 全景分割：背景pixel分类+框里mask  

![](./images/图像分割的类型.png)

### 视频目标分割
1. 通常会给定目标mask，然后求特定目标的mask

![](./images/视频目标分割.png)

### 视频实例分割
1. 根据目标检测的框，求目标的mask

![](./images/视频实例分割.png)

### 图像分割的上采样方法
* 1 双线性插值
![](./images/双线性插值.png)
`v(0,0),v(0,1),v(1,0),v(1,1)`是定义在图像坐标系下的坐标值
* 2 反池化  
![](./images/on_pooling.png)

* 3 Transpose Conv(转置卷积)
标准卷积的矩阵表示
  ![](./images/卷积操作的矩阵表示.png)
  

## 参考链接
* 1 [图像分割综述](https://aistudio.baidu.com/aistudio/education/preview/1838052)
* 2 [FCN全卷积网络详解](https://aistudio.baidu.com/aistudio/education/preview/756664)

