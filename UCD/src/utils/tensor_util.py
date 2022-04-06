"""
Helper functions to tensor operations.
"""
import torch


def padding_list_of_tensors(sequences):
    """
    :param sequences: list of tensors
    :return: all tensors in the list padded.
    """
    num_word_defns = [s.size(0) for s in sequences]
    max_len = max(num_word_defns)
    for i, tensor in enumerate(sequences):
        padding = torch.nn.ConstantPad2d((0, 0, 0, max_len-sequences[i].size(0)), 0)
        sequences[i] = torch.unsqueeze(padding(sequences[i]), 0)

    return sequences, torch.Tensor(num_word_defns)
