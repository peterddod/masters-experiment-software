import numpy as np
import torch
from src.train import evaluate_epoch
import os
from collections import Counter


def make_activation_matrix(byte_sequence, neuron_count):
    """
    Create a `torch.Tensor` matrix for the activation of each neuron in a
    neural network per dataset sample.
    """
    bytes_per_pattern = int(np.ceil(neuron_count/8))

    byte_array = [byte_sequence[i:i+bytes_per_pattern] for i in range(0,len(byte_sequence),bytes_per_pattern)]
    bit_string_array = [ "".join(list((map(lambda y: '{0:08b}'.format(y), x))))[:neuron_count] for x in byte_array ]
    int_array = [int(x) for x in list("".join(bit_string_array))]
    activation_matrix = torch.Tensor(int_array).reshape(-1,neuron_count)

    return activation_matrix


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
        _model_out.append(output.flatten(1).detach())
        return

    model.apply_forward_hook(hook)

    for data, target in dataloader:
        data = data.to(device)
        model(data)
        _output.append(torch.hstack(_model_out).detach())

    _output = torch.hstack(_output).detach()

    _output[_output!=0] = 1

    return _output


def cache_pattern_matrix(filename, input_name, data, model_cls, device):
    """
    Produces activation patterns matrix from a saved model when using a dataset and saves
    a .pt to the specifed path.
    """
    if filename==None:  return
    model_path = f'./results/{input_name}/snapshots/{filename}.pt'
    cache_path = f'./processed/{input_name}/.cache/{filename}.pt'
    model = model_cls()
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    activation_matrix = extract(model, data, device)
    torch.save(activation_matrix, cache_path)


def get_number_of_unique_patterns(activation_matrix):
    convert_func = lambda i: hash(str(i.tolist()))
    activation_matrix_hash = map(convert_func, activation_matrix)
    counter = Counter(activation_matrix_hash)

    return len(counter.keys())  


def cos_sim(a, b):
    return torch.dot(a, b)/(torch.linalg.norm(a)*torch.linalg.norm(b))


def get_similarities(m1, m2, compareFunction=cos_sim):
    if m1==None or m2==None: return [0]
    if m1.shape != m2.shape: return [0]

    output = []

    for i in range(m1.shape[0]):
        output.append(compareFunction(m1[i], m2[i]))

    return output


def load_cached_pattern(inputname, filename):
    output = None
    try:
        output = torch.load(f'./processed/{inputname}/.cache/{filename}.pt', map_location=torch.device('cpu')).detach()
    except Exception as e:
        print(e)
        pass

    return output


def load_snapshot(inputname, filename, model_cls):
    output = None
    try:
        model = model_cls()
        model.load_state_dict(torch.load(f'./results/{inputname}/snapshots/{filename}.pt', map_location=torch.device('cpu')))
        output = model

    except Exception as e:
        print(e)
        pass

    return output


def get_prev(current_filename, n, samplerate, updates_in_epoch, batch_size):
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

    return f'{epoch}_{update}'


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


def process_statistics(filename, output, test_dataloader, input_name, model_cls, at_10_filename, at_100_filename, measure_unique_patterns, device):
    """
    Processes the statistics for a particular update.

    Inputs a queue and adds a .csv string in the following order:
    - epoch
    - batch_idx
    - test_acc
    - unique_ap
    - mean_ap_sim@10
    - std_ap_sim@10
    - mean_ap_sim@100
    - std_ap_sim@100
    - mean_ap_sim@init
    - std_ap_sim@init
    - mean_ap_sim@final
    - std_ap_sim@final
    - mean_wc@10
    - std_wc@10
    - mean_wc@100
    - std_wc@100
    - mean_wc@init
    - std_wc@init
    - mean_wc@final
    - std_wc@final
    """
    if filename == None: return 

    output_measures = []

    split = filename.split('_')
    epoch = int(split[0])
    batch_idx = int(split[1])

    model = model_cls()
    model.load_state_dict(torch.load(f'./results/{input_name}/snapshots/{filename}.pt', map_location=torch.device('cpu')))
    test_acc = evaluate_epoch(model, test_dataloader, device)

    output_measures.append(str(epoch))
    output_measures.append(str(batch_idx))
    output_measures.append(str(test_acc))

    activation_matrix = torch.load(f'./processed/{input_name}/.cache/{filename}.pt').detach()

    if measure_unique_patterns: output_measures.append(str(get_number_of_unique_patterns(activation_matrix)))

    at_10 = load_cached_pattern(input_name, at_10_filename)
    sim_at_10 = get_similarities(at_10, activation_matrix)
    output_measures.append(np.mean(sim_at_10))
    output_measures.append(np.std(sim_at_10))

    at_100 = load_cached_pattern(input_name, at_100_filename)
    sim_at_100 = get_similarities(at_100, activation_matrix)
    output_measures.append(np.mean(sim_at_100))
    output_measures.append(np.std(sim_at_100))

    at_init = load_cached_pattern(input_name, 'init')
    sim_at_init = get_similarities(at_init, activation_matrix)
    output_measures.append(np.mean(sim_at_init))
    output_measures.append(np.std(sim_at_init))

    at_final = load_cached_pattern(input_name, 'final')
    sim_at_final = get_similarities(at_final, activation_matrix)
    output_measures.append(np.mean(sim_at_final))
    output_measures.append(np.std(sim_at_final))


    at_10_model = load_snapshot(input_name, at_10_filename, model_cls)
    wc_at_10 = get_mean_abs_weight_difference(model, at_10_model)
    output_measures.append(wc_at_10.mean().item())
    output_measures.append(wc_at_10.std().item())

    at_100_model = load_snapshot(input_name, at_100_filename, model_cls)
    wc_at_100 = get_mean_abs_weight_difference(model, at_100_model)
    output_measures.append(wc_at_100.mean().item())
    output_measures.append(wc_at_100.std().item())

    at_init_model = load_snapshot(input_name, 'init', model_cls)
    wc_at_init = get_mean_abs_weight_difference(model, at_init_model)
    output_measures.append(wc_at_init.mean().item())
    output_measures.append(wc_at_init.std().item())

    at_final_model = load_snapshot(input_name, 'final', model_cls)
    wc_at_final = get_mean_abs_weight_difference(model, at_final_model)
    output_measures.append(wc_at_final.mean().item())
    output_measures.append(wc_at_final.std().item())

    output(",".join([str(x) for x in output_measures]))


def delete_unused(old, new, input_name):
    [os.remove(f'./processed/{input_name}/.cache/{filename}.pt') for filename in set(old) - set(new) if filename != None]