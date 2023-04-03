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

        self.module_to_patterns_map = {}
        self.activate = nn.ReLU()

        def record(layer_name):
            def hook(model, input, output):
                output = self.activate(output.detach())
                output[output!=0] = 1
                self.module_to_patterns_map[layer_name] = output
            return hook

        self.pattern_selector.apply_forward_hook(record)

        self.model = model

        def grad_func(layer_name):
            def hook(grad):
                activation_grad = self.module_to_patterns_map[layer_name].mean(0)
                return grad*activation_grad
            return hook

        self.model.apply_hook(grad_func)

        def forward_activate(layer_name):
            def hook(model, input, output):
                return output*self.module_to_patterns_map[layer_name]
            return hook

        self.model.apply_forward_hook(forward_activate)
        

    def forward(self, x):
        with torch.no_grad():
            self.pattern_selector(x)

        x = self.model(x)
        return x
