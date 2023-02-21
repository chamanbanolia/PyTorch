## Training the Network
* The model that you built in the previous topic is not yet ready for actual prediction.
* Neural networks are universal function approximators that map input to the output. In our case, we are planning MNIST digits to their class probabilities.
* The network is trained by adjusting the weights and bias step by step until it attains reasonable accuracy.

## Model Performance
* To find the right parameters, it is necessary to compute how poorly our network is performing on real outputs by using a loss function.
* The most generic loss function we can think of is mean-squared error given by $l=\frac{1}{2n}\sum^{i=1 \ to \ n}(y_i-\hat{y_i})^2$ 
  * where n is the number of training examples $y_i$ is the true labels, and $\hat{y_i}$ are the predicted labels.
* Using this loss, we compute the gradients and adjust the parameters of our network throughout the training process.

## Criterion
* Loss functions are usually referred to as criterion which means the criteria on which we are evaluating the model.
* In our case, i.e., classifying MNIST digits which are a multi-class classification problem.
* Since our output is probabilities computed by softmax function, we will be using `cross-entropy loss` as our loss function.
* The `nn` module of torch function provides an implementation for [cross-entropy](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss) loss along with several other loss functions.

### Crieterion - Implementation
* We define the criterion for our problem as shown below.
```
# Define the loss
criterion = nn.CrossEntropyLoss()
```
Loss value for first batch of the dataset:
```
# Get our data
images, labels = next(iter(trainloader))
# Flatten images
images = images.view(images.shape[0], -1)

# Forward pass, get our logits
logits = model(images)
# Calculate the loss with the logits and the labels
loss = criterion(logits, labels)

print(loss)

output: tensor(2.2810)
```
### :bulb: Remember
* The Cross entropy loss internally computes the softmax probabilities given the linear input.
* You can skip the final softmax output in the final layer as shown below.
```
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10))
```
* For our MNIST digits problem, we will continue to use the model as defined above.

## Computing Gradients
* Once we have defined our criterion, we calculate the gradients of the parameters against the loss function.
* Using these gradients, we update our parameters at every iteration of the training step by step.
* Torch has a module, autograd, for automatically calculating the gradients of tensors.
* We can use it to calculate the gradients of all our parameters with respect to the loss.
* Autograd works by keeping track of operations performed on tensors, then going back through those operations, calculating gradients along the way.

### Defining Function
* For the sake of understanding, let's see how the PyTorch autograd works.
* Randomly initialize a tensor
```
x = torch.randn(2,2, requires_grad=True)
print(x)
Output:  tensor([[ 0.7652, -1.4550],
                [-1.2232,  0.1810]])
```
* To make sure PyTorch keeps track of operations on a tensor and calculates the gradients, you need to set requires_grad = True
  Define a function:
```
y = x**2
z = y.mean()
```
Next, we will see how to compute the gradient of x with respect to z using autograd.
### Autograd
* In order to compute the gradients on a tensor, the autograd module keeps track of functions that the tensor is being used as parameters.
* For example:
```
print(y)
output:
tensor([[1.7163, 1.1595],
              [0.3548, 0.3609]], grad_fn=<PowBackward0>)
print(z)
output:
tensor(0.8979, grad_fn=<MeanBackward1>)
```
* The `grad_fn` shows the operation that is used to create the variable.
* On using `.backward()` on z, we get derivative of x with respect to z.
```
z.backward()
##Compute gradients using autograd
print(x.grad)
tensor([[-0.6550,  0.5384],
        [-0.2978, -0.3004]])

###gradients if computed mannually
print(x/2)
tensor([[-0.6550,  0.5384],
        [-0.2978, -0.3004]], grad_fn=<DivBackward0>)
```
* You can see that gradients from autograd and the one computed manually is the same.

### Loss and Autograd Together
* In previous cards, you have seen how to compute gradients on simple function.
* However, in the actual neural network, we have lots of parameters, and auto grad function automatically computes the gradients for these parameters.
* Now, we come back to our original MNIST predictions and compute the gradients of weights and bias parameters against the loss function.

```
criterion = nn.CrossEntropyLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logits = model(images)
loss = criterion(logits, labels)

### print gradients of weights from the first layer
print('Before backward pass: \n', model[0].weight.grad)

loss.backward()

print('After backward pass: \n', model[0].weight.grad) 

output:
Before backward pass: 
 None
After backward pass: 
 tensor([[ 0.0028,  0.0028,  0.0028,  ...,  0.0028,  0.0028,  0.0028],
        [-0.0016, -0.0016, -0.0016,  ..., -0.0016, -0.0016, -0.0016....
```
* `loss.backward()` computes the gradients with respect to weight and bias parameters against the loss function.
* The gradients are accessed through `.grade()` function.

## Optimizer Function
* There is one last piece we need to start training, an optimizer that we'll use to update the weights with the gradients.
* Now, we have the gradients obtained from the autograd function.
* Next step is to have an optimizer to update the weights with gradients.
* To do this, we use PyTorch's `optim` [package](https://pytorch.org/docs/stable/optim.html). For example, we can utilize stochastic gradient descent with `optim.SGD`.
```
from torch import optim

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(params = model.parameters(), lr=0.01)
```
* The `params` parameter looks for the parameters that need to be updated during the training.

## Updating Weights
* To update the parameters using the gradients use .step() function on the optimizer, you defined in the previous card.
```
optimizer.step()
```
* The above step is equivalent to perm parameter updation using the formula $W=W-\alpha * \frac{\partial L}{\partial W}$ is the learning rate and L is the loss function and W is the weights of the network.
* The `.step()` function updates the parameters by taking the gradients computed by `loss.backward()`.
* To view the updated parameters
```
print('Updated weights of first layer - ', model[0].weight)
```
## Training the Network
* Now, we have the Model, criterion, and the optimizer defined.
* To train the model:
  * Make a forward pass through the network.
  * Use the network output to calculate the loss.
  * Perform a backward pass through the network with loss.backward() to calculate the gradients.
  * Take a step with the optimizer to update the weights.
  * We perform the above steps for a number of iterations known as epochs such the loss is close to zero.

### Training the Netowrk - Code
* Training the network for 5 epochs
```
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        ### Training pass
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        
                optimizer.step()
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")
```
* If you run this code, you can see that the loss values decrease at each iteration.
* The gradients get accumulated at each iteration; hence it is necessary to clear the gradients by running optimizer.zero_grad().
