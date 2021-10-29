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

## 代码构建resnet50网络结构
```python
        # self.block_list = []
        self.stage_list = []
        if layers >= 50:
            # `block`表示每一个卷积模块，分别是`conv2_x`、`conv3_x`、`conv4_x`、`conv5_x`卷积模块
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                # `depth[block]`表示每个卷积模块重复的次数
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)

                    ###############################################################################
                    # Add dilation rate for some segmentation tasks, if dilation_dict is not None.
                    dilation_rate = dilation_dict[
                        block] if dilation_dict and block in dilation_dict else 1

                    # Actually block here is 'stage', and i is 'block' in 'stage'
                    # At the stage 4, expand the the dilation_rate if given multi_grid
                    if block == 3:
                        dilation_rate = dilation_rate * multi_grid[i]
                    ###############################################################################
                    # BottleneckBlock模块包含了三个`ConvBNLayer`卷积层以及是否需要短连接操作
                    # `ConvBNLayer` 包含了卷积、标准化、激活操作
                    bottleneck_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BottleneckBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block] * 4,
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0
                            and dilation_rate == 1 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            dilation=dilation_rate,
                            data_format=data_format))

                    block_list.append(bottleneck_block)
                    shortcut = True
                self.stage_list.append(block_list)
        else:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    dilation_rate = dilation_dict[block] \
                        if dilation_dict and block in dilation_dict else 1
                    if block == 3:
                        dilation_rate = dilation_rate * multi_grid[i]

                    basic_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BasicBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block],
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 \
                                and dilation_rate == 1 else 1,
                            dilation=dilation_rate,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            data_format=data_format))
                    block_list.append(basic_block)
                    shortcut = True
                self.stage_list.append(block_list)

        self.pretrained = pretrained
        self.init_weight()
```
* 构建后的resnet50网络结构

<details><summary>展开/收起</summary>
<pre><code>

ResNet_vd(
  (conv1_1): ConvBNLayer(
    (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
    (_conv): Conv2D(3, 32, kernel_size=[3, 3], stride=[2, 2], padding=1, data_format=NCHW)
    (_batch_norm): SyncBatchNorm(num_features=32, momentum=0.9, epsilon=1e-05)
    (_act_op): Activation(
      (act_func): ReLU()
    )
  )
  (conv1_2): ConvBNLayer(
    (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
    (_conv): Conv2D(32, 32, kernel_size=[3, 3], padding=1, data_format=NCHW)
    (_batch_norm): SyncBatchNorm(num_features=32, momentum=0.9, epsilon=1e-05)
    (_act_op): Activation(
      (act_func): ReLU()
    )
  )
  (conv1_3): ConvBNLayer(
    (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
    (_conv): Conv2D(32, 64, kernel_size=[3, 3], padding=1, data_format=NCHW)
    (_batch_norm): SyncBatchNorm(num_features=64, momentum=0.9, epsilon=1e-05)
    (_act_op): Activation(
      (act_func): ReLU()
    )
  )
  (pool2d_max): MaxPool2D(kernel_size=3, stride=2, padding=1)
  (bb_0_0): BottleneckBlock(
    (conv0): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(64, 64, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=64, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv1): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(64, 64, kernel_size=[3, 3], padding=1, data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=64, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv2): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(64, 256, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=256, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
    (short): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(64, 256, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=256, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
  )
  (bb_0_1): BottleneckBlock(
    (conv0): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(256, 64, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=64, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv1): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(64, 64, kernel_size=[3, 3], padding=1, data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=64, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv2): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(64, 256, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=256, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
  )
  (bb_0_2): BottleneckBlock(
    (conv0): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(256, 64, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=64, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv1): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(64, 64, kernel_size=[3, 3], padding=1, data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=64, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv2): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(64, 256, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=256, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
  )
  (bb_1_0): BottleneckBlock(
    (conv0): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(256, 128, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=128, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv1): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(128, 128, kernel_size=[3, 3], stride=[2, 2], padding=1, data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=128, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv2): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(128, 512, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=512, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
    (short): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(256, 512, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=512, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
  )
  (bb_1_1): BottleneckBlock(
    (conv0): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(512, 128, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=128, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv1): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=128, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv2): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(128, 512, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=512, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
  )
  (bb_1_2): BottleneckBlock(
    (conv0): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(512, 128, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=128, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv1): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=128, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv2): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(128, 512, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=512, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
  )
  (bb_1_3): BottleneckBlock(
    (conv0): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(512, 128, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=128, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv1): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(128, 128, kernel_size=[3, 3], padding=1, data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=128, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv2): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(128, 512, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=512, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
  )
  (bb_2_0): BottleneckBlock(
    (conv0): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(512, 256, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=256, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv1): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(256, 256, kernel_size=[3, 3], dilation=[2, 2], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=256, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv2): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(256, 1024, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=1024, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
    (short): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(512, 1024, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=1024, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
  )
  (bb_2_1): BottleneckBlock(
    (conv0): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(1024, 256, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=256, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv1): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(256, 256, kernel_size=[3, 3], dilation=[2, 2], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=256, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv2): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(256, 1024, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=1024, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
  )
  (bb_2_2): BottleneckBlock(
    (conv0): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(1024, 256, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=256, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv1): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(256, 256, kernel_size=[3, 3], dilation=[2, 2], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=256, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv2): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(256, 1024, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=1024, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
  )
  (bb_2_3): BottleneckBlock(
    (conv0): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(1024, 256, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=256, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv1): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(256, 256, kernel_size=[3, 3], dilation=[2, 2], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=256, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv2): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(256, 1024, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=1024, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
  )
  (bb_2_4): BottleneckBlock(
    (conv0): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(1024, 256, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=256, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv1): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(256, 256, kernel_size=[3, 3], dilation=[2, 2], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=256, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv2): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(256, 1024, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=1024, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
  )
  (bb_2_5): BottleneckBlock(
    (conv0): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(1024, 256, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=256, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv1): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(256, 256, kernel_size=[3, 3], dilation=[2, 2], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=256, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv2): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(256, 1024, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=1024, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
  )
  (bb_3_0): BottleneckBlock(
    (conv0): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(1024, 512, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=512, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv1): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(512, 512, kernel_size=[3, 3], dilation=[4, 4], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=512, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv2): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(512, 2048, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=2048, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
    (short): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(1024, 2048, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=2048, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
  )
  (bb_3_1): BottleneckBlock(
    (conv0): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(2048, 512, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=512, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv1): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(512, 512, kernel_size=[3, 3], dilation=[4, 4], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=512, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv2): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(512, 2048, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=2048, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
  )
  (bb_3_2): BottleneckBlock(
    (conv0): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(2048, 512, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=512, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv1): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(512, 512, kernel_size=[3, 3], dilation=[4, 4], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=512, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv2): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(512, 2048, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=2048, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
  )
)
</code></pre>
</details>  

## resnet50网络结构解析
* `conv1_*`包含了三个卷积操作  
  `bb_0_0`,`bb_0_1`,`bb_0_2` 
  `bb_1_0`,`bb_1_1`,`bb_1_2`,`bb_1_3`
  `bb_2_0`,`bb_2_1`,`bb_2_2`,`bb_2_3`,`bb_2_4`,`bb_2_5`
  `bb_3_0`,`bb_3_1`,`bb_3_2`  
分别做了`BottleneckBlock`操作，BottleneckBlock模块包含了三个`ConvBNLayer`卷积层以及是否需要短连接操作
 ```text
BottleneckBlock(
    (conv0): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(2048, 512, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=512, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv1): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(512, 512, kernel_size=[3, 3], dilation=[4, 4], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=512, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation(
        (act_func): ReLU()
      )
    )
    (conv2): ConvBNLayer(
      (_pool2d_avg): AvgPool2D(kernel_size=2, stride=2, padding=0)
      (_conv): Conv2D(512, 2048, kernel_size=[1, 1], data_format=NCHW)
      (_batch_norm): SyncBatchNorm(num_features=2048, momentum=0.9, epsilon=1e-05)
      (_act_op): Activation()
    )
  )
``` 
## 参考链接
* 1 [resnet文章](https://arxiv.org/pdf/1512.03385.pdf)
* 2 [paddlex之resnet_rd.py](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/paddlex/paddleseg/models/backbones/resnet_vd.py)

