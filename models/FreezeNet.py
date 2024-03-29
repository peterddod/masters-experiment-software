from models.modules import Activator
from torch import nn
import torch

from utils import resetseed


class FreezeNet(nn.Module):
    """
    Takes structure from pattern selector. Only trains model.
    Model should be identical to pattern selector but without an activation function.
    """
    def __init__(self, model_cls, pattern_selector=None, seed=0):
        super(FreezeNet, self).__init__()

        self.model_cls = model_cls

        resetseed(seed)
        self.pattern_selector = self.model_cls()
        self.pattern_selector.load_state_dict(pattern_selector)

        self.activation_matrix_queue = []
        self.activate = nn.ReLU()

        def record_activation_pattern(model, input, output):
            output = self.activate(output.detach())
            output[output!=0] = 1
            self.activation_matrix_queue.append(output)

        self.pattern_selector.apply_forward_hook(record_activation_pattern)

        self.model = None

        self.frozen = False

        self.active_model = self.pattern_selector


    def get_activation_matrix(self):
        return self.activation_matrix_queue.pop(0)
    

    def parameters(self):
        return self.active_model.parameters()
    

    def train(self):
        self.active_model.train()


    def eval(self):
        self.active_model.eval()


    def freeze(self):
        self.model = self.model_cls()
        self.model.load_state_dict(self.pattern_selector.state_dict())
        self.pattern_selector.eval()
        self.model.set_activate(Activator, get_activation_matrix=self.get_activation_matrix)
        self.active_model = self.model
        self.frozen = True


    def apply_forward_hook(self, hook):
        return self.active_model.apply_forward_hook(hook)


    def state_dict(self):
        return self.active_model.state_dict()


    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                self.pattern_selector(x)
        x = self.active_model(x)
        return x
    
    def to(self, device):
        return self.active_model.to(device)
    

    def __call__(self, x):
        return self.forward(x)
