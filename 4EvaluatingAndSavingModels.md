## Model Evaluation
* The performance of the model depends on how well it predicts the labels for the input that haven't been used for training.
* In the following cards, we will measure the accuracy of our model to perform on test data.

## Test Loss
* To find the test loss, we follow the similar procedure that you used for finding the training loss.
* We iterate each batch of our test data and predict the output though forward pass.
```
test_loss = 0
for images, labels in testloader:
    images = images.view(images.shape[0], -1)
    probs = model(images)
    test_loss += criterion(probs, labels)
print(test_loss/len(testloader))
output: 
tensor(0.2960, grad_fn=<DivBackward0>)
```
## Test Accuracy
* Test accuracy tells us the fraction of output that is predicted right out of all the available predictions.
* The test labels are actual values of the digits in the image, and our model output is a set of linear outputs by each neuron in the final layer.
* Follow the below code to compute the test accuracy.
```
accuracy = 0
optimizer.zero_grad()
for images, labels in testloader:
    images = images.view(images.shape[0], -1)
    probs = model(images)
    
    top_p, top_class = probs.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy += torch.mean(equals.type(torch.FloatTensor))
        print("Accuracy ",  (accuracy/len(testloader) ** 100).item())
        Output: 
        Accuracy: 91.4112
```
* probs.topk(1, dim = 1) picks the max value and its index from the output probs across each row.
## Saving the Model
* It would be impractical to retrain a model whenever you want to use it.
* Instead, it would be better to save the model and load it back whenever we want either for resuming the training or for predictions.
* It is possible to extract the weights and biases of each layer of the network and load it back when required.

## Model's State Dict
* All the model's parameters are stored in model's state_dict and to access these parameters, we use model.state_dict() as shown below.
```
print(model.state_dict())

OrderedDict([('fc1.weight', tensor([[-0.0032,  0.0181,  0.0224,  ...,  0.0246, -0.0026,  0.0332],  [ 0.0112,  0.0103,  0.0311,  ..., -0.0105,  0.0289,  0.0336]...),
('fc1.bias', tensor([-0.0044, -0.0352,  0.0307,  0.0264, -0.0305, -0.0040, -0.0213, -0.0318,   0.0321,  0.0157,  0.0298, -0.0156, -0.0145,  0.0216,  0.0154,  0.0287,...)...
```
* We use the parameters from the state_dict to save the trained weights and bias of our model.

## Saving Model to File
* Before we save our model, it is good to define the model configurations into a dictionary.
```
checkpoint = {'input_size': 784,
              'hidden_size': [128, 64],
              'output_size': 10,
              'state_dict': model.state_dict()}
```
* Once we have the configurations as dictionary, use `torch.save()` to save the dictionary into a file.
```
torch.save(checkpoint, 'checkpoint.pth')
```
The second argument is the file name usually you wish to save the model usually as `.pth` or `.pt` extension.

## Loading the Saved Model
* In order to load back the configurations of a model from the file, use torch.load as shown below.
```
checkpoint = torch.load('checkpoint.pth')
```
* This return the same dictionary you defined while saving the model.
* Next, define the model.
```
model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(checkpoint["input_size"], checkpoint["hidden_size"][0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(checkpoint["hidden_size"][0], checkpoint["hidden_size"][1])),
                      ('relu2', nn.ReLU()),
                      ('output', nn.Linear(checkpoint["hidden_size"][1], checkpoint["output_size"]))]))
```
* Once we have defined our model, use model.load_state_dict() to load the trained weights and bias as
```
model.load_state_dict(checkpoint["state_dict"]) 
```
