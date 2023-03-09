import numpy as np
import torch


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
    Find the file names required to process a batch_idx in file_process.py
    """
    buffer = max(comparisons) + batch_size

    storage_start_point = batch_idx * batch_size
    processing_end_point = storage_start_point + buffer

    output_list = []

    while storage_start_point < processing_end_point:
        epoch = (storage_start_point*samplerate) // updates_in_epoch + 1
        update = (storage_start_point*samplerate) % updates_in_epoch

        output_list.append(f'{epoch}_{update}')

        storage_start_point += 1

    print(output_list)

    return output_list
