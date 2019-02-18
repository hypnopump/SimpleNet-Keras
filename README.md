# SimpleNet-Keras

SimpleNet paper implementation in keras (a [pretrained](#pretrained-model)) model is provided). This is the implementation for the **SimpleNet** model architecture described in [this paper](https://arxiv.org/ftp/arxiv/papers/1608/1608.06037.pdf) for the CIFAR10 dataset. The original implementation in Caffe can be found [here](https://github.com/Coderx7/SimpleNet).

## Results

The paper describes a top accuracy of 95.32% on Cifar10 (Krizhevsky & Hinton, 2009), although I just managed to get it past 94% following all the indications in the paper and their training and model logs.

It takes a little bit long to train and, although it's proposed as a Simple Network, it can be too big for some GPUs with less than 3GB of memory. For that reason, I recommend to train them on Amazon Web Services (AWS) or any other cloud platform such as Alibaba Cloud or Google Cloud. 

## Custom activations

In order to build the model with a custom activation function, just create it like the following example:

```python
def custom_act(x):
	return 1.3*K.sigmoid(x)*x
```
Then the only remaining thing is to pass the function as the `act` parameter the the `create_model()` function:

```python
model = create_model(act=custom_act)
```

## Pretrained Model

The model was trained as described in the `simplenet_training.ipynb` notebook and achieved 94% accuracy with 125 epochs. The learning rate adjustment performed in the original model (Caffe implementation) was not respected. Here the model was only trained for 125 instead of 200 and learning rates were different (the change was done aiming to speed up training). It has been saved using keras `model.save()` function. 
To load the model in your own code: 

```python
from keras.models import load_model

model = load_model("simplenet_trained_model.h5")
```

## Meta
 
* **Author's GitHub Profile**: [Eric Alcaide](https://github.com/EricAlcaide/)
* **Twitter**: [@eric_alcaide](https://twitter.com/eric_alcaide)
* **LinkedIn**: [@ericalcaide](https://www.linkedin.com/in/ericalcaide/)
* **Email**: ericalcaide1@gmail.com
