from torch import nn
import torch


class FreezeNet(nn.Module):
    """
    Takes structure from pattern selector. Only trains model.
    Model should be identical to pattern selector but without an activation function.
    """
    def __init__(self, pattern_selector, model):
        super(FreezeNet, self).__init__()

        self.pattern_selector = pattern_selector
        self.pattern_selector.eval()

        self.activation_matrix_queue = []
        self.activate = nn.ReLU()

        def record_activation_pattern(model, input, output):
            output = output.detach()
            output[output!=0] = 1
            self.activation_matrix_queue.append(output)

        self.pattern_selector.apply_forward_hook(record_activation_pattern)

        self.model = model


    def get_activation_matrix(self):
        return self.activation_matrix_queue.pop(0)


    def forward(self, x):
        with torch.no_grad():
            self.pattern_selector(x)
        x = self.model(x)
        return x
