import torch
from config import SIM_FUNC_MAP
from src.similarities import get_similarities
from src.train import evaluate_epoch
import os
from collections import Counter


def get_filenames_to_process(samplerate, comparisons, batch_size, batch_idx, updates_in_epoch, total_number_of_updates):
    """
    Find the file names required to process a batch_idx in file_process.py.
    Returns None item in list if batch exceeds updates_in_epochs, 
    and returns None once processing window passess all updates.
    """
    buffer = max(comparisons) + batch_size

    storage_start_point = batch_idx * batch_size
    processing_start_point = storage_start_point + max(comparisons)
    processing_end_point = storage_start_point + buffer

    if processing_start_point >= total_number_of_updates//samplerate:
        return None

    output_list = []

    while storage_start_point < processing_end_point:
        if storage_start_point >= total_number_of_updates//samplerate:
            output_list.append(None)
        else:
            epoch = (storage_start_point*samplerate) // updates_in_epoch + 1
            update = (storage_start_point*samplerate) % updates_in_epoch

            output_list.append(f'{epoch}_{update}')

        storage_start_point += 1

    return output_list


def extract(model, dataloader, device):
    """
    Extract activation patterns from model using data and return as
    torch tensor.
    """
    model.eval()

    _output = []
    _model_out = []

    def hook(model, input, output):
        output = output.flatten(1).detach()
        _model_out.append(output)
        return

    handles = model.apply_forward_hook(hook)

    for data, target in dataloader:
        data = data.to(device)
        model(data)
        _output.append(torch.hstack(_model_out).detach())
        _model_out = []

    _output = torch.vstack(_output).detach()

    _output[_output!=0] = 1

    for handle in handles: handle.remove()

    return _output


def cache_pattern_matrix(filename, data, model_cls, snapshots_path, cache_path, device):
    """
    Produces activation patterns matrix from a saved model when using a dataset and saves
    a .pt to the specifed path.
    """
    if filename==None:  return
    model_path = f'{snapshots_path}{filename}.pt'
    cached_pattern_path = f'{cache_path}{filename}.pt'
    model = model_cls()
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    activation_matrix = extract(model, data, device)
    torch.save(activation_matrix, cached_pattern_path)


def get_number_of_unique_patterns(activation_matrix):
    """
    Counts the number of nuique patterns in an activation_matrix.
    Works poorly for non-small networks (bigger than lenet5).
    """
    convert_func = lambda i: hash(str(i.tolist()))
    activation_matrix_hash = map(convert_func, activation_matrix)
    counter = Counter(activation_matrix_hash)

    return len(counter.keys())  


def load_cached_pattern(filename, cache_path):
    """
    Load a cached pattern from the cache path with a particular filename.
    """
    output = None
    try:
        output = torch.load(f'{cache_path}{filename}.pt', map_location=torch.device('cpu')).detach()
    except Exception as e:
        print(e)
        pass

    return output


def load_snapshot(filename, model_cls, snapshots_path):
    """
    Load snapshot of model `model_cls` from `snapshots_path` with filename `filename`.
    `filename` should exclude .pt extension.
    """
    output = None
    try:
        model = model_cls()
        model.load_state_dict(torch.load(f'{snapshots_path}{filename}.pt', map_location=torch.device('cpu')))
        output = model

    except Exception as e:
        print(e)
        pass

    return output


def get_prev(current_filename, n, samplerate, updates_in_epoch):
    """
    Find the previous n filename if exists, otherwise return None.
    """
    if current_filename == None: return 

    split = current_filename.split('_')
    epoch = int(split[0])
    batch_idx = int(split[1])

    current_update = (((epoch-1) * updates_in_epoch) + batch_idx)
    next_update = current_update - n*samplerate

    if next_update < 0:
        return None

    epoch = (next_update) // updates_in_epoch + 1
    update = (next_update) % updates_in_epoch

    return f'{int(epoch)}_{int(update)}'


def get_mean_abs_weight_difference(model1, model2):
    """
    Calculates the mean elementwise absolute difference in weights between two identically structured models.
    """
    if model1==None or model2==None: return torch.Tensor([0])

    diff_list = []

    model2_modules = [*model2.modules()]

    for i, module in enumerate(model1.modules()):
        try:
            weight1 = module.weight.data
            weight2 = model2_modules[i].weight.data

            diff_list.append(torch.abs(weight1 - weight2).flatten())
        except Exception as e:
            pass

        try:
            weight1 = module.bias.data
            weight2 = model2_modules[i].bias.data

            diff_list.append(torch.abs(weight1 - weight2).flatten())
        except:
            pass

    return torch.cat(diff_list)


def get_epoch(**kwargs):
    return int(kwargs['filename'].split('_')[0])


def get_batch_idx(**kwargs):
    return int(kwargs['filename'].split('_')[1])


def get_test_acc(**kwargs):
    test_acc = evaluate_epoch(kwargs['model'], kwargs["test_loader"], kwargs["device"])
    return test_acc


def get_unique_patterns(**kwargs):
    return get_number_of_unique_patterns(kwargs['activation_matrix'])


def get_weight_change(at, **kwargs):
    if at.isdigit():
        compare_at = int(at)/kwargs['samplerate']
        at = get_prev(kwargs['filename'], compare_at, kwargs['samplerate'], kwargs['updates_in_epoch'])
    
    if at == None:
        return [0,0]

    at_model = load_snapshot(at, kwargs['model_cls'], kwargs['snapshots_path'])
    result = get_mean_abs_weight_difference(kwargs['model'], at_model)

    return result


def get_similarity(at, similarity, **kwargs):
    if at.isdigit():
        compare_at = int(at)/kwargs['samplerate']
        at = get_prev(kwargs['filename'], compare_at, kwargs['samplerate'], kwargs['updates_in_epoch'])

    if at == None:
        return [0,0]

    at_pattern = load_cached_pattern(at, kwargs['cache_path'])
    result = get_similarities(
        at_pattern, 
        kwargs['activation_matrix'], 
        compareFunction = SIM_FUNC_MAP[similarity],
    )

    return result


def get_similarity_per_layer(at, similarity, **kwargs):
    if at.isdigit():
        compare_at = int(at)/kwargs['samplerate']
        at = get_prev(kwargs['filename'], compare_at, kwargs['samplerate'], kwargs['updates_in_epoch'])

    if at == None:
        return [0,0]
    
    at_pattern = load_cached_pattern(at, kwargs['cache_path'])
    
    result_dict = {}

    for layer_index, layer_start_index in enumerate(kwargs['layer_indexes']):
        layer_start_index = int(layer_start_index)

        if layer_index+1 >= len(kwargs['layer_indexes']):
            layer_end_index = kwargs['activation_matrix'].shape[1]
        else:
            layer_end_index = kwargs['layer_indexes'][layer_index+1]

        layer_at_pattern = at_pattern[:,layer_start_index:layer_end_index]
        current_pattern = kwargs['activation_matrix'][:,layer_start_index:layer_end_index]

        result = get_similarities(
            layer_at_pattern, 
            current_pattern, 
            compareFunction = SIM_FUNC_MAP[similarity],
        )

        result_dict[layer_index] = result

    return result_dict


def process(measure, **kwargs):
    """
    Take a given measure and return the desired value
    """
    PROCESS_MAP = {
        'epoch': get_epoch,
        'batch_idx': get_batch_idx,
        'test_acc': get_test_acc,
        'unique_patterns': get_unique_patterns,
        'wc': get_weight_change,
        'sim': get_similarity,
        'sim_pl': get_similarity_per_layer,
    }

    if '@' in measure:
        measure_split = measure.split('@')
        measure = measure_split[0]
        params = measure_split[1].split('_')
    else:
        params = []

    result = PROCESS_MAP[measure](*params, **kwargs)

    return result


def delete_unused(old, new, cache_path):
    """
    Removes unused cached patterns from cache_path.
    """
    [os.remove(f'{cache_path}{filename}.pt') for filename in set(old) - set(new) if filename != None]