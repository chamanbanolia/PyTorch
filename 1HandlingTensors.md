# Tensors

* Data is a primary fuel to the neural network.
* In most of the deep learning frameworks, the data is enclosed in a container called Tensors.
* PyTorch has an exclusive class calledtorch.Tensor() used to initialize data as tensors.
* Defining data as tensors provide numerous inherent functions such as matrix multiplication, reshaping, and computing gradients.

## Initializing Tensors
The below code snippets shows various ways of randomly initializing tensors:
* Generating one dimensional tensor with 10 elements randomly generated from a standard normal distribution:
```
x = torch.randn(10)
print(x)
print(x.size())

output:
tensor([-0.4441,  1.9954, -0.3540, -0.2576,  1.0282,  0.0635, -0.7485, -2.0920,
       -1.0542, -0.7134])

torch.Size([10])
```
* Similarly, you will be able to generate an n-dimensional tensors as:
```
torch.randn(2,3)   ### a 2-d tensor
torch.randn(2,3,6) ### a 3-d tensor
```
* In general, the sequence of integers as arguments defines the shape of the tensor.
### Initializing Tensors with Data
* You have seen how to initialize a tensor with random values.
* Initializing with random values comes into the picture when you need to initialize the weights and bias of the neural network to start the learning the process.
* We can also initialize the Tensors with known values as shown in the below code:
```
x = torch.Tensor([[1,2], [3,4]])
print(x)
print(x.size())

output:
tensor([[1., 2.],
        [3., 4.]])

torch.Size([2, 2])
```
* This kind of initialization is useful when you need to pass the training data to the network as tensors.
### 3-D Tensor
* 3-dimensional tensors are especially useful when you are dealing with image data.
* The dimension of an image file will usually be represented as `image_length * image_width * no. of channels`.
* In the case of convolution neural networks, it is necessary to convert the image data to tensors.

### Image to Tensor
![image](https://user-images.githubusercontent.com/44977122/220179298-202dc66b-7930-4961-ada7-3930a6b720f1.png)
* The below code snippet shows one of the several ways to convert image data into tensors.
```
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
img = np.array(Image.open("bird.jpeg").resize((224,224)))
img_tensor = torch.Tensor(img)
plt.imshow(img)
print("shape of image tensor {}".format(img_tensor.shape))
```
* In the above example, we first read the image file (you can take an image of your own) using the Python PIL package, resize the first two dimension of the image as 224 x 224.
* Later, we retrieve the pixel intensity values as NumPy array.
* Finally, we typecast the numpy array as tensor using torch.Tensor() constructor.

### Slicing Tensor
![image](https://user-images.githubusercontent.com/44977122/220179957-adf547e6-8915-4bf4-8a05-5a2e276e80ec.png)

* You can also extract a part of the tensor, say only the first channel of the image through slicing as shown in the below code.
```
channen_0 = img_tensor[:, : , 0]
print("shape of first channel of image data:{}".format(channen_0.shape))
plt.imshow(channen_0)
```
* You can also slice the tensor using the start and end index as shown below:
```
img_slice = img_tensor[25:175,60:130,0]
print("shape of the sliced image.{}".format(img_slice.size()))
plt.imshow(img_slice)
```
### Reshaping Tensor
* To reshape an existing tensor to new shape use [.view()](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view) on the tensor.
```
x = torch.Tensor([[1,2], [3,4]])
print("before reshaping", x)
Output: 
before reshaping tensor([[1., 2.],
                        [3., 4.]])

x = x.view(1,-1)
print("after reshaping", x)
Output: After reshaping: tensor([[1., 2., 3., 4.]])
```
### Useful Operations
* [torch.topk()](http://https//pytorch.org/docs/stable/torch.html#torch.topk) returns the top k values and their index along the given dimension.

```
arr = torch.Tensor([[10,20,40],[80,60, 70]])
arr.topk(k = 1, dim = 1)
output:
### top one value along the column
(tensor([[40.],[80.]]), 
tensor([[2],  [0]]))

###top one value along the row
arr.top(k = 1, dim = 0)
output:
(tensor([[80., 60., 70.]]), tensor([[1, 1, 1]]))
```
* Use torch.Tensor.item() to get a scalar number from a tensor containing a single value.
```
torch.Tensor([10]).item()
output:
10.0
```
* On any operations, the function name followed by '_' says the tensors are modified inplace.
```
x = torch.Tensor([[1,2], [3,4]])
###multiply each element by 2
x.mul_(2)
print(x)
Output:
tensor([[2., 4.],
        [6., 8.]])
```
### Torch to NumPy
* Since Torch is basically built on NumPy you can convert you Tensors to NumPy object and vice versa.
* Converting Tensors to Numpy
```
x = torch.Tensor([[1,2], [3,4]])
print(type(x))
output: <class 'torch.Tensor'>

y = x.numpy()
print(type(y))
output: <class 'numpy.ndarray'>
```
* The memory is shared between the Numpy array and Torch tensor, so if you change the values in-place of one object, the other will change as well.
```
x.mul_(2)  ### 
print(x)
output:
tensor([[2., 4.],
        [6., 8.]])

print(y)
output:
array([[2., 4.],
       [6., 8.]], dtype=float32)
```
* You can convert back the numpy array back to Tensor object as shown below:
```
z = torch.from_numpy(x)
print(type(z))
output:  <class 'torch.Tensor'>
```
### Tensors on GPU
* Computing vectorized operations are way faster in GPU compared to CPU.
* For this, it is necessary to shift your data (or tensors) from CPU to GPU.
* PyTorch provides a handy function [.cuda()](https://pytorch.org/docs/stable/cuda.html) to move tensors from CPU to GPU
```
### To check if GPU instance is vailable in you system or not
torch.cuda.is_available()  

Output:
True  ##(False otherwise)
```
### Moving Tensors to GPU
* Torch provides a handy function .cuda() to move tensors to GPU.
* On any tensor object calling .cuda() on will shift the tensors to GPU provided the GPU instance is available.
* Computation time to perform matrix multiplication on CPU

```
a = torch.rand(10000,10000)
b = torch.rand(10000,10000)

import time
start = time.time()
a.matmul(b)
end = time.time()
print("{} seconds".format(end - start))

Output: 
12.40355 seconds
```
* Computation time after moving the tensors to GPU
```
#Move the tensors to GPU
a = a.cuda()
b = b.cuda()

start = time.time()
a.matmul(b)
end = time.time()
print("{} seconds".format(end - start))

output: 
0.000011 seconds
```
