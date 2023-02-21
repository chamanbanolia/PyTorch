# Building Feed Forward Neural Networks in PyTorch
* The following are the necessary steps performed to build a neural network in PyTorch.
1. Load dataset
2. Make dataset iterable
3. Create and initiate Model class
4. Initiate Loss class
5. Initiate Optimizer class
6. Train Model
* Now, let us see how to build a simple feedforward network by following the above steps.

## Dataset
![image](https://user-images.githubusercontent.com/44977122/220249318-b6e727ab-e919-425a-9f6b-a622377cca36.png)
* MNIST dataset is a collection of grayscale images of handwritten digits as shown in the example above.
* Each number in the picture is 28 x 28 pixels.
* We will be using this dataset to build and train a network to recognize any the handwritten digits.

## Loading the Dataset
* To download the MNIST data set, we will use `torchvision` package as shown in the below code.
```
from torchvision import datasets
from torchvision import transforms
trainset = datasets.MNIST(root='./data',  train=True, transform=transforms.ToTensor(),
                            download=True)
```

* The above code downloads the training dataset into the data folder of the current directory (a new folder will be created if not already present).
* The `transforms.ToTensor()` converts images or NumPy array to Tensor.
## Generate Mini-Batch
* When we train the network using a mini batch, we feed the data in batches in each training epoch.
* For example, we feed 64 images at a time and then compute the gradients at the end of each batch.
* `torch.utils.data.Dataloader()` function transforms the tensors into iterable mini batches as shown in the below code snippet.
```
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
```
* The batch_size the number of images(or data) we get in one iteration from the data loader and pass through our network often called as a batch.
* `shuffle = True` specifies to shuffle the data every time you start iterating the trainloader again.

## Visualizing Sample Image
* Now, we have the data loaded as iterable batches.
* Follow the below code snippet to visualize the first image from the first batch provided by trainloader, which we defined in the previous card.
```
import matplotlib.pyplot as plt
data_iter = iter(trainloader)
images, labels = data_iter.next()

print(type(images))   ### <class 'torch.Tensor'>
print(images.shape)  ### torch.Size([64, 1, 28, 28])
print(labels.shape)  ### torch.Size([64])

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
```
* As you have observed, the images from the first batch is of shape [64,1,28,28], which means in a batch, there are 64 images each of 28 * 28 pixels with single channel.

## Feed Forward Network Architecture
![image](https://user-images.githubusercontent.com/44977122/220250077-f9c36e54-72cd-4486-92e4-fc6fc935b096.png)
* Now, let's try to build a network to recognize the MNIST digits as per the specifications in the above image.
* Since we want our network to predict one among ten classes, we have ten nodes in the final classification layer.

## Building the Network
* PyTorch provides a module `nn` that makes building networks much simpler.
* The below code defines a feedforward network as per the specifications outlined in the previous card.
```
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128,64)
        self.output = nn.Linear(64,10)
        
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.softmax(self.output(x),dim = 1)
        return x

model = Network()
```
### Create and Initialize Model Class
* `class Network(nn.Module)`: Here we are defining a class named Network. Adding `super().__init__()` inherits all the properties of `nn.Module` that tracks the architecture and provides lots of useful methods and attributes.
  * `self.hidden1 = nn.Linear(784, 128)`: This line creates a module for a linear transformation, $xW+b$, with 784 inputs and 128 outputs and assigns it to `self.hidden1`. The module automatically creates the weight and bias tensors, which, we'll use in the `forward` method.
  * Similarly, below code creates another two layers of 64 units and ten units.
    ```
    self.hidden2 = nn.Linear(128,64)
    self.output = nn.Linear(64,10)
    ```
### Forward Function
* Referring to the code in the previous card, in the same class, we have defined another method `forward`.
* PyTorch networks created with `nn.Module` must have a **forward** method defined. It takes in a tensor x and passes it through the operations you defined in the *_init_* method.
* The `torch.nn.functional` module imported as F provides useful methods for various network operations such as activations, normalization, and dropout.
* It doesn't matter what you name the variables here, as long as the inputs and outputs of the operations match the network architecture you want to build.
* The order in which you define things in the *_init_* method doesn't matter, but you'll need to sequence the operations correctly in the forward method.   

## Initializing Model Class
* Once you have defined the model `Network`, it is initialized as shown below.
```
model = Network()

print(model)

Output:
Network(
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=10, bias=True)
)
```
* printing model object outputs layer specifications.
* The layer weights are initialized automatically. You can view the parameters of each layer as shown in the below example.
```
print(model.fc1.weight)  ## weight parameters of first layer
print(model.fc1.bias)     ## bais parameter of fist layer
```
* We can manually change the weight and bias parameters
```
model.fc1.bias.data.fill_(0) ### initilize all bias values of first layer to zeros inplace
model.fc1.weight.data.normal_(std=0.01)  ##intialize weights with normal distribution with standard devaition 0.01
```
## Define Model - nn.Sequential
* You can also define the model using nn.Sequential class of PyTorch, which is a much more convenient and straight forward.
```
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),  
                   nn.ReLU(),
                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                   nn.ReLU(),
                   nn.Linear(hidden_sizes[1], output_size),
                   nn.Softmax(dim=1))
print(model)
```
* Here is another way of defining the model, where you pass an OrderedDict to name the individual layers and operations.
```
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))

print(model)
print(model[0])   ### accessing first layer by index
print(model.fc1) ### accessing first layer by layer name
```
## Flattening
* Since this is a feed-forward network, it cannot accept a 2D tensor of 28 * 28-pixel image.
* So it is necessary to flatten the image such that pixels in each row are stacked one above the other, so that final size is [1,784].
* For a given tensor, you can use either .resize() or .view() to reshape the tensors.
```
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Resize images into a 1D vector, new shape is (batch size, color channels, image pixels).
images.resize_(64, 1, 784)
# or images.resize_(images.shape[0], 1, 784) to automatically get batch size
```
## Forward Pass
* Once we have flattened the images, we are ready to pass these tensors into the model as shown in the below example.
```
# Obtain images and labels from first batch
images, labels = next(iter(trainloader))
#Flatten the images
images.resize_(images.shape[0], 1, 784)
## Forward pass a image through the network after resizing
ps = model.forward(images[0,:])
```
## Output
![image](https://user-images.githubusercontent.com/44977122/220252039-8103c07a-db85-42ef-aee9-5b481aa910fd.png)
* As you can see, the plotting probability score shows that the probability of the input images belonging to a particular class is equally likely.
* This is because we have not trained the network yet and all model weights are random.
* In the next topic, we will see how to train and validate our model that you have already defined.

