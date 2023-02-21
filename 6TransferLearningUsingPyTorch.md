## Transfer Learning
* The basic premise of transfer learning is to reuse the model trained on a large dataset to a smaller subset.
* Following are the necessary steps followed in transfer learning
  * Load the pre-trained model
  * Freeze the parameters in lower layers of the model
  * Add custom classifier with several layers of trainable parameters
  * Train the classifier layers on the training data available for the task.
  * Fine-tune parameters and unfreeze more layer if needed.

## Loading Pre-Trained Model
* PyTorch's [torchvision.models](https://pytorch.org/docs/stable/torchvision/models.html) provides a variety of pre-trained models that you can download and customize it to your dataset.
* The example below shows how to download VGG16 model using torchvision.
```
from torchvision import models
model = models.vgg16(pretrained=True)
print(model)
```
* The above code will initialize the VGG16 model along with the pre-trained weights.
* If `pretrained = False` then the weights would be randomized which are meant to be trained from scratch.
* `print(model)` shows the specification of each layer of the VGG16 model

## Understanding the Model
* The VGG16 model that we downloaded earlier has had two parts, one is the convolution layers for feature extractions, and other is the classification layer which is a set fully connected layer.
* These layers are defined as OrderedDict which can be accessed as shown below
* View convolution layers
```
print(model.features)
```
* View classification layers
```
print(model.classifier)
```
## Freezing the Layers
* This model has over 130 million parameters, but we’ll train only the very last few fully-connected layers.
* So in the course of training, we keep the weights of convolution layers undisturbed and only update the weights of the classification layer.
* To prevent the weights of convolution layers from being updated, we turn off their gradients as shown below
```
# Freeze training for all "features" layers
for param in model.features.parameters():
    param.requires_grad = False
```
## Customizing the Model
* The VGG model is trained over image-net dataset to classify over a thousand classes.
* Since we are dealing with the cifar10 dataset, we have to customize the model to classifier over ten classes.
* To do this, we have to replace the final layer in the group of classification layers to have ten nodes.
* The final layer receives the portion of input from the layers you are not changing and produce an appropriate number of outputs.

## Modify FC Layer
* Current final FC layer of our model
```
print(model.classifier[6])
output: 
Linear(in_features=4096, out_features=1000, bias=True)
```
* Replacing final layer
```
n_classes = 10
in_features = model.classifier[6].in_features

 

last_layer = nn.Linear(in_features, n_classes)
model.classifier[6] = last_layer

 

print(model.classifier)
output:

 

Sequential(
  (0): Linear(in_features=25088, out_features=4096, bias=True)
  (1): ReLU(inplace)
  (2): Dropout(p=0.5)
  (3): Linear(in_features=4096, out_features=4096, bias=True)
  (4): ReLU(inplace)
  (5): Dropout(p=0.5)
  (6): Linear(in_features=4096, out_features=10, bias=True)
)
```
