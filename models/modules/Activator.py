from torch import nn
import torch


class Activator(nn.Module):
    def __init__(self, get_activation_matrix):
        super(Activator, self).__init__()

        self.get_activation_matrix = get_activation_matrix
        self.func = None


    def forward(self, x):
        activation_matrix = self.get_activation_matrix()

        class ActivatorFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                return input * activation_matrix
            
            @staticmethod
            def backward(ctx, grad_output):
                return grad_output * activation_matrix
            
        self.func = ActivatorFunction()
        x = self.func.apply(x)
        return x