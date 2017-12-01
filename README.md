# SimpleNet-Keras

SimpleNet paper implementation in keras. This is the implementation for the **SimpleNet** model architecture described in [this paper](https://arxiv.org/ftp/arxiv/papers/1608/1608.06037.pdf) for the CIFAR10 dataset. The original implementation in Caffe can be found [here](https://github.com/Coderx7/SimpleNet).

## Results

The paper describes a top accuracy of 95.32% on Cifar10 (Krizhevsky & Hinton, 2009), although I just managed to get it past 93% following all the indications in the paper and their training and model logs.

It takes a little bit long to train and, although it's proposed as a Simple Network, it can be too big for some GPUs. For that reason I recommend to train them on Amazon Web Services (AWS) or any other cloud platform as Alibaba Cloud or Google Cloud. 