import numpy as np
import torch
from utils import FileWriter
from collections.abc import Iterable


class MeasureCollector:
    def __init__(self, measures, similarities, number_of_layers, path):
        self._determine_all_measures(measures, similarities, number_of_layers)
        self.file = FileWriter(path, ",".join(self.measure_headers))

        self.current_id = -1
        self.results = []


    def _determine_all_measures(self, measures, similarities, number_of_layers):
        # Determine network similarity measures
        sim_measures = [*filter(lambda x: 'sim' in x, measures)]
        non_sim_measures = [*filter(lambda x: 'sim' not in x, measures)]

        for i, measure in enumerate(sim_measures):
            sim_measures[i] = [*map(lambda sim: f'{measure}_{sim}', similarities)]

        sim_measures = np.ravel(sim_measures)
        
        self.measures = [*non_sim_measures, *sim_measures]

        # Determine per_layer similarity measures
        sim_pl_measures = [*filter(lambda x: 'sim_pl' in x, self.measures)]
        non_sim_pl_measures = [*filter(lambda x: 'sim_pl' not in x, self.measures)]

        for i, measure in enumerate(sim_pl_measures):
            sim_pl_measures[i] = [*map(lambda layer_idx: f'{measure}_{layer_idx}', range(number_of_layers))]

        sim_pl_measures = np.ravel(sim_pl_measures)
        
        self.measure_headers = [*non_sim_pl_measures, *sim_pl_measures]

        # Determine std's and means for number sets
        at_measures = [*filter(lambda x: '@' in x, self.measure_headers)]
        non_at_measures = [*filter(lambda x: '@' not in x, self.measure_headers)]

        for i, measure in enumerate(at_measures):
            at_measures[i] = [*map(lambda op: f'{measure}_{op}', ['mean','std'])]

        at_measures = np.ravel(at_measures)

        self.measure_headers = [*non_at_measures, *at_measures]


    def next(self):
        self.current_id += 1

        if self.current_id >= len(self.measures):
            self.current_id = -1
            return None

        return self.measures[self.current_id]
    

    def add(self, data):
        if type(data) == dict:
            for value in data.values():
                if isinstance(value, torch.Tensor): value = value.numpy()
                self.results.append(np.mean(value))
                self.results.append(np.std(value))
        elif isinstance(data, Iterable):
            if isinstance(data, torch.Tensor): data = data.numpy()
            self.results.append(np.mean(data))
            self.results.append(np.std(data))
        else:
            self.results.append(data)


    def write(self):
        for i, result in enumerate(self.results): self.results[i] = str(result)

        self.file(",".join(self.results))
        self.results = []

