from collections import OrderedDict
from torch import nn

class TestModelAFFC(nn.Module):
    def __init__(self):
        super(TestModelAFFC, self).__init__()

        self.model = nn.Sequential(OrderedDict([
            ('flat',nn.Flatten()),
            ('lin1',nn.Linear(28*28,24)),
            ('relu1',nn.ReLU()),
            ('lin2',nn.Linear(24,24)),
            ('relu2',nn.ReLU()),
            ('lin3',nn.Linear(24,24)),
            ('relu3',nn.ReLU()),
            ('lin4',nn.Linear(24,10)),
        ]))
        

    def forward(self, x):
        return self.model(x)
    
    
    def apply_forward_hook(self, func):
        self.model.lin1.register_forward_hook(func('lin1'))
        self.model.lin2.register_forward_hook(func('lin2'))
        self.model.lin3.register_forward_hook(func('lin3'))