# Duke-ECE590-07-pj14-NAS
Neural Architecture Search based on NASBench-101
Arthor: Wenxin Xu, Dawei Xi

This is a README describing the general functions for each file. The detailed steps are described as comments inside each file.

### Main functions
#### NAS_Search_RS&EA&EST.ipynb
This notebook contains three parts:
Part 1, for loading the NASBench-101;
Part 2, realize three different search straties: random search, evolutionary algorithm and estimator-based search over selected search space;
Part 3, plotting and obtain the architecture from dataset.
Cited from https://github.com/googleresearch/nasbench
### Model Train.ipynb
In this notebook, we run three kinds of dataprocessing, the model is the same
Part1, auto augmentation policy with cutout;
Part2, only cutout method;
Part3, without cutout and auto augmentation policy.

### Supported functions
#### cutout.py
Realize the cutout function to process images.
Cited from https://github.com/uoguelph-mlrg/Cutout
#### CIFAR10Policy.py:
Realize the function of auto augmentation policy for CIFAR10
Cited from https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py
#### operation.py:
Realize the basic operation: conv1x1-bn-relu, conv3x3-bn-relu, downsampling, global average
#### model.py 
Realize the architecture of the cell we searched and construct the model in the paper, every stage has 3 cells and then downsampling
#### utils.py:
Realize a function
Train(net, lr, decay, momentum, device, trainloader, testloader, epoch) to train the model.

The checkpoint files exceed the uploading limitation and is thus submitted via Sakai.  
model_nas_cutout_auto.pt is the outcome of Part1  
model_nas_plain.pt is the outcome of Part3
