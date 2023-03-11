import torch
from torch import nn
from .BinaryPathTree import BinaryPathTree


"""
A path analyser for neural networks. Init, then call analyse
with a pytorch dataloader and model.
TODO: add list of layer sizes as structure
"""
class PathAnalyser:
    def __init__(self):
        self.structure = None
        self.path_tree = BinaryPathTree()

    """
    analyse the pathways created in a neural network
    params
        f: how many steps in samples between recording number of leaf nodes (unique paths), when None only returnspaths from full dataset
        n: terminate after this many samples, when None defaults to number of samples in dataset
    """
    def analyse(self, model, data_loader, n=None, f=None, device=torch.device('cpu')):
        model.eval()

        if n==None:
            n = len(data_loader)
    
        leafs = []

        weight_idx = []
        skip_idx = []
        current_idx = 0

        for i, module in enumerate(model.modules()):
            if isinstance(module, (nn.Flatten, nn.LogSoftmax)):
                continue
            elif isinstance(module, (nn.Linear, nn.LazyLinear)):
                current_idx = len(weight_idx)
                weight_idx.append(i)
            elif isinstance(module, (nn.ReLU)):
                weight_idx[current_idx] = i
            else:
                skip_idx.append(i)
        
        sample_count = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device).to(torch.float32), target.to(device).to(torch.float32)

                outputs = []

                for i, module in enumerate(model.modules()):
                    if i in skip_idx:
                        continue

                    data = module(data)

                    if i in weight_idx:
                        outputs.append(torch.flatten(data,1))

                path = torch.hstack(outputs)
                
                path = path.squeeze()

                path[path!=0] = 1
                self.path_tree.add(list(path))
                
                sample_count += 1
                
                if f!=None:
                    if sample_count%f==0:
                        leafs.append(self.path_tree.get_number_of_leafs())
                    
                if sample_count==n:
                    break

        if f==None:
            return self.path_tree.get_number_of_leafs()

        return leafs