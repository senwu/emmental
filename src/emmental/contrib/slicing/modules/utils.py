import torch.nn.functional as F


def ce_loss(module_name, intermediate_output_dict, Y, active):
    return F.cross_entropy(
        intermediate_output_dict[module_name][0][active], (Y.view(-1) - 1)[active]
    )


def output(module_name, intermediate_output_dict):
    return F.softmax(intermediate_output_dict[module_name][0], dim=1)
