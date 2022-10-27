import random
from tqdm import trange
import crypten
import numpy as np
import torch
import torchvision
import os
from crypten.nn.loss import CrossEntropyLoss

from src.layer import Dense, ReLU

critertion = CrossEntropyLoss()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed = 42

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_mnist() -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    """
    load MNIST data
    :return: x_train, y_train, x_test, y_test
    """
    # download the dataset from torchvision
    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )

    # create torch.Tensor objects for the training and test data
    X_train: torch.Tensor = train_dataset.data.reshape(-1, 28 * 28).float()/255
    y_train = train_dataset.targets
    # Standardize the test data
    X_test = test_dataset.data.reshape(-1, 28 * 28).float()/255
    y_test = test_dataset.targets

    return X_train, y_train, X_test, y_test





def softmax_crossentropy_with_logits(logits: crypten.CrypTensor, reference_answers):
    # Compute crossentropy from logits[batch,n_classes] and ids of correct answers
    one_hot = torch.nn.functional.one_hot(reference_answers, 10)
    # logits_for_answers = logits[torch.arange(len(logits)),reference_answers]

    # xentropy = - logits_for_answers + torch.log(torch.sum(torch.exp(logits),axis=-1))
    xentropy = critertion(logits, one_hot)

    return xentropy


def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    # Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers
    # ones_for_answers = torch.zeros_like(logits)
    # ones_for_answers[torch.arange(len(logits)),reference_answers] = 1
    one_hot = torch.nn.functional.one_hot(reference_answers, 10)

    softmax = logits.softmax(1)  # torch.exp(logits) / torch.exp(logits).sum(axis=-1,keepdims=True)

    return (softmax - one_hot) / logits.shape[0]


def forward(network, X):
    # Compute activations of all network layers by applying them sequentially.
    # Return a list of activations for each layer.

    activations = []
    input = X
    # Looping through each layer
    for l in network:
        activations.append(l.forward(input))
        # Updating input to last layer output
        input = activations[-1]

    assert len(activations) == len(network)
    return activations


def predict(network, X):
    # Compute network predictions. Returning indices of largest Logit probability
    logits = forward(network, X)[-1]
    return logits.argmax(axis=-1)


def train(network, X, y):
    # Train our network on a given batch of X and y.
    # We first need to run forward to get all layer activations.
    # Then we can run layer.backward going from last to first layer.
    # After we have called backward for all layers, all Dense layers have already made one gradient step.

    # Get the layer activations
    layer_activations = forward(network, X)
    layer_inputs = [X] + layer_activations  # layer_input[i] is an input for network[i]
    logits = layer_activations[-1]

    # Compute the loss and the initial gradient
    loss = softmax_crossentropy_with_logits(logits, y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits, y)

    # Propagate gradients through the network
    # Reverse propogation as this is backprop
    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]

        loss_grad = layer.backward(layer_inputs[layer_index], loss_grad)  # grad w.r.t. input, also weight updates
    return loss  # torch.mean(loss)

def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

def run_crypten_mnist(num_epochs, learning_rate, batch_size):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    X_train, y_train, X_test, y_test = load_mnist()
    network = []
    network.append(Dense(X_train.shape[1], 100, learning_rate))
    network.append(ReLU())
    network.append(Dense(100, 200, learning_rate))
    network.append(ReLU())
    network.append(Dense(200, 10, learning_rate))
    for epoch in range(num_epochs):
        for x_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size=batch_size, shuffle=True):
            x_batch = crypten.cryptensor(x_batch, device=DEVICE)  # encrypt the features

            loss = train(network, x_batch, y_batch)
            print(loss.get_plain_text())
