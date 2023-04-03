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

        def record_activation_pattern(layer_name):
            def hook(model, input, output):
                output = self.activate(output.detach())  # ReLU not added in model as hook added to lin/conv layers directly
                output[output!=0] = 1
                self.module_to_patterns_map[layer_name] = output
            return hook

        self.pattern_selector.apply_forward_hook(record_activation_pattern)

        self.model = model

        def grad_func(layer_name):
            def hook(module, grad_in, grad_out):
                activation_grad = self.module_to_patterns_map[layer_name]
                grad_out_mod = tuple([grad_out[0]*activation_grad])
                return grad_out_mod
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
