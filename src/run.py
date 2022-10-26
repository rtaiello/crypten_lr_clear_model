from typing import Dict, List, Tuple

import torch
import torchvision
import numpy as np
import crypten
import random
import os

from tqdm import trange
from crypten.nn.loss import CrossEntropyLoss
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
    x_train: torch.Tensor = train_dataset.data.reshape(-1, 28 * 28).float()
    x_train_mean: float = x_train.mean()
    x_train_std: float = x_train.std()
    # Standardize the training data
    x_train_norm = (x_train - x_train_mean) / x_train_std
    y_train = train_dataset.targets
    # Standardize the test data
    x_test = test_dataset.data.reshape(-1, 28 * 28).float()
    x_test_norm = (x_test - x_train_mean) / x_train_std
    y_test = test_dataset.targets

    return x_train_norm, y_train, x_test_norm, y_test

def initialize(num_inputs, num_classes) -> Dict[str, torch.Tensor]:
    """
    initialize the parameters
    :param num_inputs:
    :param num_classes:
    :return:
    """
    w: torch.Tensor = torch.rand(num_inputs, num_classes, device=DEVICE)
    b: torch.Tensor = torch.rand(1, num_classes, device=DEVICE)

    param: Dict[str, torch.Tensor] = {"w": w, "b": b}  # (10*784)  # (10*1)
    return param


def train_crypten(
    param: Dict[str, torch.Tensor],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    criterion: crypten.nn.loss,
    verbose: bool = False,
    num_classes: int = 10
) -> Tuple[Dict[str, torch.Tensor], List[float]]:
    """
    Train the model with the given parameters
    :param param:
    :param x_train:
    :param y_train:
    :param num_epochs:
    :param batch_size:
    :param learning_rate:
    :return:
    """
    losses: List[float] = []
    current_loss = 0
    for _ in trange(num_epochs, desc="Epoch"):
        # select the random sequence of training set
        rand_indices: List[int] = np.random.choice(
            x_train.shape[0], x_train.shape[0], replace=False
        )
        num_batches: int = int(x_train.shape[0] / batch_size)
        for batch in trange(num_batches, desc="Batch"):
            index = rand_indices[batch * batch_size : (batch + 1) * batch_size]
            x_batch = x_train[index]  # (batch_size, num_inputs)
            y_batch = y_train[index]  # (batch_size, 1)
            x_batch = crypten.cryptensor(x_batch, device=DEVICE)  # encrypt the features
            logits = x_batch.matmul(param["w"]) + param["b"]  # (num_classes, 1)

            # https://codesti.com/issue/facebookresearch/CrypTen/278
            y_one_hot = torch.nn.functional.one_hot(y_batch, num_classes)
            if verbose:
                loss = criterion(logits, y_one_hot)
                current_loss = loss.get_plain_text()
                print(current_loss)
            # https://aaronkub.com/2020/02/12/logistic-regression-with-pytorch.html#why-logistic-regression
            activation = logits.softmax(1)
            w_gradients = (
                -x_batch.transpose(0, 1).matmul(-activation + y_one_hot) / batch_size
            )  # (num_inputs, num_classes)
            b_gradients = -(-activation + y_one_hot).mean(0)
            param["w"] -= learning_rate * w_gradients.get_plain_text()
            param["b"] -= learning_rate * b_gradients.get_plain_text()
        losses.append(current_loss)
    return param, losses


def evaluation(
    param: Dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor
) -> None:
    test_predictions = torch.argmax(
        torch.softmax(x.matmul(param["w"]) + param["b"], dim=1), dim=1
    )
    test_accuracy = torch.sum(test_predictions == y).float() / y.shape[0]
    print(f"Accuracy: {test_accuracy}")


def run_lr_mnist(num_epochs, learning_rate, batch_size):
    crypten.init()
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    criterion = CrossEntropyLoss()

    x_train, y_train, x_test, y_test = load_mnist()
    x_train = x_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    x_test = x_test.to(DEVICE)
    y_test = y_test.to(DEVICE)
    num_inputs = x_train.shape[1]
    num_classes = len(torch.unique(y_train))
    param = initialize(num_inputs, num_classes)
    param, _ =  train_crypten(param=param, x_train=x_train, y_train=y_train,num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, criterion=criterion,num_classes=num_classes, verbose=False)
    print("Evaluation training over encrypted features")
    evaluation(param, x_test, y_test)


