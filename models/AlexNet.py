from torch import nn
from utils import he_init


class AlexNet(nn.Module):
    def __init__(self, activate=nn.ReLU):
        super(AlexNet, self).__init__()

        self.activate = activate
        self.reset_activation_func()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96),
        )
        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
        )
        self.block3 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
        )
        self.block6 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2304, 1024),
        )
        self.block7 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
        )

        self.fc_out = nn.Linear(1024, 10)

        self.block1.apply(he_init)
        self.block2.apply(he_init)
        self.block3.apply(he_init)
        self.block4.apply(he_init)
        self.block5.apply(he_init)
        self.block6.apply(he_init)
        self.block7.apply(he_init)
        self.fc_out.apply(he_init)
        

    def forward(self, x):
        x = self.block1(x)
        x = self.af1(x)
        x = self.block2(x)
        x = self.af2(x)
        x = self.block3(x)
        x = self.af3(x)
        x = self.block4(x)
        x = self.af4(x)
        x = self.block5(x)
        x = self.af5(x)
        x = self.block6(x)
        x = self.af6(x)
        x = self.block7(x)
        x = self.af7(x)

        x = self.fc_out(x)

        return x
    

    def reset_activation_func(self, **kwargs):
        self.af1 = self.activate(**kwargs)
        self.af2 = self.activate(**kwargs)
        self.af3 = self.activate(**kwargs)
        self.af4 = self.activate(**kwargs)
        self.af5 = self.activate(**kwargs)
        self.af6 = self.activate(**kwargs)
        self.af7 = self.activate(**kwargs)


    def set_activate(self, activate, **kwargs):
        self.activate = activate
        self.reset_activation_func(**kwargs)


    def apply_forward_hook(self, func):
        handles = []
        handles.append(self.af1.register_forward_hook(func))
        handles.append(self.af2.register_forward_hook(func))
        handles.append(self.af3.register_forward_hook(func))
        handles.append(self.af4.register_forward_hook(func))
        handles.append(self.af5.register_forward_hook(func))
        handles.append(self.af6.register_forward_hook(func))
        handles.append(self.af7.register_forward_hook(func))
        return handles