import torch
from .utils import *
from torch import nn


class PatternCollector:
    def __init__(self, model, data):
        self.model = model
        self.model.eval()
        self.data = data.data.to(torch.float32)

        self.patternSet = set() 
        self.patternMatrix = None
        
        self.similarity_vector_last_step = None

        # find where to get APs from
        self.weight_idx = []
        self.skip_idx = []

        for i, module in enumerate(model.modules()):
            if isinstance(module, (nn.Flatten, nn.LogSoftmax, nn.Linear, nn.LazyLinear)):
                continue
            elif isinstance(module, (nn.ReLU)):
                self.weight_idx.append(i)
            else:
                self.skip_idx.append(i)

        # get APs with datatset
        for datapoint in self.data:
            datapoint = datapoint.reshape([1,*datapoint.shape])

            # get AP
            pattern = self.getActivationPattern(datapoint)

            # add to pattern matrix
            if (self.patternMatrix==None):
                self.patternMatrix = pattern
            else:
                self.patternMatrix = torch.vstack([self.patternMatrix, pattern])

            # update pattern set
            self.patternSet.add(hash(str(pattern)))

    def getModel(self):
        return self.model

    def getActivationPattern(self, data):
        outputs = []

        for i, module in enumerate(self.model.modules()):
            if i in self.skip_idx:
                continue

            data = module(data)

            if i in self.weight_idx:
                outputs.append(torch.flatten(data,1))

        path = torch.hstack(outputs).detach()
        
        path = path.squeeze()

        path[path!=0] = 1

        return path


    def update(self): 
        self.model.eval()
        before = self.patternMatrix.clone()

        self.patternSet = set()

        for i, datapoint in enumerate(self.data):
            datapoint = datapoint.reshape([1,*datapoint.shape])

            # get AP
            pattern = self.getActivationPattern(datapoint)

            # add pattern to matrix
            self.patternMatrix[i] = pattern

            # update pattern set
            self.patternSet.add(hash(str(pattern)))

        # get mean cosine similarity between before and patternMatrix
        self.similarity_vector_last_step = vector_similarity(before.detach(), self.patternMatrix.detach(), compareFunction=cos_sim)
        before = None

    def getNumberOfUniquePatterns(self):
        return len(self.patternSet)

    def getMeanCosineOfPatternsToLastStep(self):
        return np.mean(self.similarity_vector_last_step)

    def getStdCosineOfPatternsToLastStep(self):
        return np.std(self.similarity_vector_last_step)

    def getNumberOfChangedPatternsFromLastStep(self):
        return