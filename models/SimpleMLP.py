from torch import nn
from utils import he_init


class SimpleMLP(nn.Module):
    def __init__(self, activate=nn.ReLU):
        super(SimpleMLP, self).__init__()

        self.activate = activate
        self.reset_activation_func()

        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(28*28,24)
        self.lin2 = nn.Linear(24,24)
        self.lin3 = nn.Linear(24,24)
        self.lin4 = nn.Linear(24,10)

        self.apply(he_init)
        

    def forward(self, x):
        x = self.flat(x)
        x = self.lin1(x)
        x = self.af1(x)
        x = self.lin2(x)
        x = self.af2(x)
        x = self.lin3(x)
        x = self.af3(x)
        x = self.lin4(x)
        return x
    

    def reset_activation_func(self, **kwargs):
        self.af1 = self.activate(**kwargs)
        self.af2 = self.activate(**kwargs)
        self.af3 = self.activate(**kwargs)
        self.af4 = self.activate(**kwargs)


    def set_activate(self, activate, **kwargs):
        self.activate = activate
        self.reset_activation_func(**kwargs)
    

    def apply_forward_hook(self, func):
        handles = []
        handles.append(self.af1.register_forward_hook(func))
        handles.append(self.af2.register_forward_hook(func))
        handles.append(self.af3.register_forward_hook(func))
        return handles