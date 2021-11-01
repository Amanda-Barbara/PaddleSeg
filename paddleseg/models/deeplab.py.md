## deeplab.py 代码解析

## align_corners角点对齐操作
* [对输入数据进行上采样操作时的角点对齐方法](https://blog.csdn.net/wangweiwells/article/details/101820932)
![](data/images/align_corners.png)

## Decoder流程
```python
Decoder(nn.Layer):
    """
    Decoder module of DeepLabV3P model

    Args:
        num_classes (int): The number of classes.
        in_channels (int): The number of input channels in decoder module.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 align_corners,
                 data_format='NCHW'):
        super(Decoder, self).__init__()

        self.data_format = data_format
        self.conv_bn_relu1 = layers.ConvBNReLU(
            in_channels=in_channels,
            out_channels=48,
            kernel_size=1,
            data_format=data_format)

        self.conv_bn_relu2 = layers.SeparableConvBNReLU(
            in_channels=304,
            out_channels=256,
            kernel_size=3,
            padding=1,
            data_format=data_format)
        self.conv_bn_relu3 = layers.SeparableConvBNReLU(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1,
            data_format=data_format)
        self.conv = nn.Conv2D(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=1,
            data_format=data_format)

        self.align_corners = align_corners

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv_bn_relu1(low_level_feat)
        if self.data_format == 'NCHW':
            low_level_shape = paddle.shape(low_level_feat)[-2:]
            axis = 1
        else:
            low_level_shape = paddle.shape(low_level_feat)[1:3]
            axis = -1
        x = F.interpolate(
            x,
            low_level_shape,
            mode='bilinear',
            align_corners=self.align_corners,
            data_format=self.data_format)
        x = paddle.concat([x, low_level_feat], axis=axis)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.conv(x)
        return x
```
![](data/images/Decoder.png)


## 参考链接
* 1 

