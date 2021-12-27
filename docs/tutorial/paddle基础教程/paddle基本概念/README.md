# paddle基本概念

## Tensor的操作
索引和切片
您可以通过索引或切片方便地访问或修改 Tensor。Paddle 使用标准的 Python 索引规则与 Numpy 索引规则，  
与 Indexing a list or a string in Python类似。具有以下特点：

1. 基于 0-n 的下标进行索引，如果下标为负数，则从尾部开始计算

2. 通过冒号 : 分隔切片参数 start:stop:step 来进行切片操作，其中 start、stop、step 均可缺省

* 访问 Tensor(针对1-D Tensor，则仅有单个轴上的索引或切片：)
```python
ndim_1_tensor = paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
print("Origin Tensor:", ndim_1_tensor.numpy())

print("First element:", ndim_1_tensor[0].numpy())
print("Last element:", ndim_1_tensor[-1].numpy())
print("All element:", ndim_1_tensor[:].numpy())
print("Before 3:", ndim_1_tensor[:3].numpy())
print("From 6 to the end:", ndim_1_tensor[6:].numpy())
print("From 3 to 6:", ndim_1_tensor[3:6].numpy())
print("Interval of 3:", ndim_1_tensor[::3].numpy())
print("Reverse:", ndim_1_tensor[::-1].numpy())
```
输出结果：
```text
Origin Tensor: [0 1 2 3 4 5 6 7 8])
First element: [0]
Last element: [8]
All element: [0 1 2 3 4 5 6 7 8]
Before 3: [0 1 2]
From 6 to the end: [6 7 8]
From 3 to 6: [3 4 5]
Interval of 3: [0 3 6]
Reverse: [8 7 6 5 4 3 2 1 0]
```
* 针对2-D及以上的 Tensor，则会有多个轴上的索引或切片：
[a,b,c]表示一行中有三列，每列元素分别是a,b,c的一维向量  
[[a],[a],[a],[a]]表示四行1列的向量，其中[]后跟着的,表示换行，
```python
ndim_2_tensor = paddle.to_tensor([[0, 1, 2, 3],
                                  [4, 5, 6, 7],
                                  [8, 9, 10, 11]])
print("Origin Tensor:", ndim_2_tensor.numpy())
print("First row:", ndim_2_tensor[0].numpy())
print("First row:", ndim_2_tensor[0, :].numpy())
print("First column:", ndim_2_tensor[:, 0].numpy())
print("Last column:", ndim_2_tensor[:, -1].numpy())
print("All element:", ndim_2_tensor[:].numpy())
print("First row and second column:", ndim_2_tensor[0, 1].numpy())
```
输出结果：
```text
Origin Tensor: [[ 0  1  2  3]
                [ 4  5  6  7]
                [ 8  9 10 11]]
First row: [0 1 2 3]
First row: [0 1 2 3]
First column: [0 4 8]
Last column: [ 3  7 11]
All element: [[ 0  1  2  3]
              [ 4  5  6  7]
              [ 8  9 10 11]]
First row and second column: [1]
```


## 参考链接
* 1 [Tensor基本概念](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/basic_concept/tensor_introduction_cn.html)
