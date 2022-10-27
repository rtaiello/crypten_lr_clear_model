from abc import ABC, abstractmethod
import crypten
import torch
import math


class Layer(ABC):
    # code from https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9
    def __init__(self)->None:
        # Here we can initialize layer parameters (if any) and auxiliary stuff.
        # A dummy layer does nothing
        pass

    @abstractmethod
    def forward(self, input: crypten.CrypTensor) -> crypten.CrypTensor:
        # Takes input data of shape [batch, input_units], returns output data [batch, output_units]
        pass

    @abstractmethod
    def backward(self, input: crypten.CrypTensor, grad_output: crypten.CrypTensor) -> crypten.CrypTensor:
        # Performs a backpropagation step through the layer, with respect to the given input.

        # To compute loss gradients w.r.t input, we need to apply chain rule (backprop):

        pass


class ReLU(Layer):
    def __init__(self)->None:
        # ReLU layer simply applies elementwise rectified linear unit to all inputs
        super().__init__()

    def forward(self, input: crypten.CrypTensor) -> crypten.CrypTensor:
        # Apply elementwise ReLU to [batch, input_units] matrix
        relu_forward = input.relu()  # torch.maximum(torch.zeros_like(input),input)
        return relu_forward

    def backward(self, input:crypten.CrypTensor, grad_output: crypten.CrypTensor) -> crypten.CrypTensor:
        # Compute gradient of loss w.r.t. ReLU input

        relu_grad = input > 0
        return grad_output * relu_grad


class Dense(Layer):
    def __init__(self, input_units: int, output_units: int, learning_rate: float=0.1)->None:
        # A dense layer is a layer which performs a learned affine transformation:
        # f(x) = <W*x> + b
        super().__init__()
        self.learning_rate = learning_rate
        self.weights = torch.normal(mean=torch.zeros((input_units, output_units)),
                                    std=math.sqrt(2 / (input_units + output_units)))
        self.biases = torch.zeros(output_units)

    def forward(self, input: crypten.CrypTensor) -> crypten.CrypTensor:
        # Perform an affine transformation:
        # f(x) = <W*x> + b

        # input shape: [batch, input_units]
        # output shape: [batch, output units]

        return input.matmul(self.weights) + self.biases

    def backward(self, input:crypten.CrypTensor, grad_output: crypten.CrypTensor) -> crypten.CrypTensor:
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = grad_output.matmul(self.weights.T)

        # compute gradient w.r.t. weights and biases
        grad_weights = input.transpose(0, 1).matmul(grad_output)
        grad_biases = grad_output.mean(axis=0) * input.shape[0]

        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape

        # Here we perform a stochastic gradient descent step.
        self.weights = self.weights - self.learning_rate * grad_weights.get_plain_text()
        self.biases = self.biases - self.learning_rate * grad_biases.get_plain_text()

        return grad_input
