import numpy as np


def get_layer_start_indexes(model, dataloader, device):
    """
    Get the starting neuron for each layer in network as list.
    """
    model.eval()

    layer_indexes = [0]
    layer_sizes = []

    def hook(model, input, output):
        output = output.flatten(1).detach()
        layer_sizes.append(output.shape[1])
        layer_indexes.append(np.sum(layer_sizes))
        return

    handles = model.apply_forward_hook(hook)

    for data, target in dataloader:
        data = data.to(device)
        model(data)
        break

    for handle in handles: handle.remove()

    return layer_indexes[:-1]