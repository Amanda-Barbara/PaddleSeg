## resnet_vd.py 代码解析

## resnet网络结构
![](data/images/resnet_arch.png)
```python
if layers == 18:
    depth = [2, 2, 2, 2]
elif layers == 34 or layers == 50:
    depth = [3, 4, 6, 3]
elif layers == 101:
    depth = [3, 4, 23, 3]
elif layers == 152:
    depth = [3, 8, 36, 3]
elif layers == 200:
    depth = [3, 12, 48, 3]
num_channels = [64, 256, 512, 1024
    ] if layers >= 50 else [64, 64, 128, 256]
num_filters = [64, 128, 256, 512]
```
`depth`列表中的数值分别表示`conv2_x`、`conv3_x`、`conv4_x`、`conv5_x`卷积模块的个数

## dilation rate 操作
```text
# 编码器配置，采用ASPP架构，pooling + 1x1_conv + 三个不同尺度的空洞卷积并行, concat后1x1conv
# ASPP_WITH_SEP_CONV：默认为真，使用depthwise可分离卷积，否则使用普通卷积
# OUTPUT_STRIDE: 下采样倍数，8或16，决定aspp_ratios大小
# aspp_ratios：ASPP模块空洞卷积的采样率
```

```python
dilation_dict = None
if output_stride == 8:
    dilation_dict = {2: 2, 3: 4}
    # 下采样的倍数为8时，分别对`conv4_x`与`conv5_x`的卷积模块的空洞卷积的采样率设置为2和4
elif output_stride == 16:
    dilation_dict = {3: 2}
    # 下采样的倍数为16时，`conv5_x`卷积模块的空洞卷积的采样率设置为2
###############################################################################
# Add dilation rate for some segmentation tasks, if dilation_dict is not None.
dilation_rate = dilation_dict[
    block] if dilation_dict and block in dilation_dict else 1

# Actually block here is 'stage', and i is 'block' in 'stage'
# At the stage 4, expand the the dilation_rate if given multi_grid
if block == 3:
    dilation_rate = dilation_rate * multi_grid[i]
    # 如果是`conv5_x`的卷积模块，则需要对空洞卷积的采样率乘以`multi_grid[i]`系数
###############################################################################
```

## 参考链接
* 1 [resnet文章](https://arxiv.org/pdf/1512.03385.pdf)