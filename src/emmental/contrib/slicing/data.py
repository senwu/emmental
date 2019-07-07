import logging

from emmental.contrib.slicing.slicing_function import slicing_function

logger = logging.getLogger(__name__)


def add_slice_labels(task, dataloaders, slice_func_dict):
    """A function to extend dataloader by adding slice indicator and predictor
    labels.
    """
    # Add base slice if needed
    if "base" not in slice_func_dict.keys():
        slice_func_dict["base"] = base_slice

    for dataloader in dataloaders:
        labels = dataloader.dataset.Y_dict[dataloader.task_to_label_dict[task.name]]
        for slice_name, slice_func in slice_func_dict.items():
            indicators = slice_func(dataloader.dataset)
            slice_ind_name = f"{task.name}_slice:ind_{slice_name}"
            slice_pred_name = f"{task.name}_slice:pred_{slice_name}"

            pred_labels = indicators * labels
            ind_labels = indicators
            ind_labels[ind_labels == 0] = 2

            # Update slice indicator and predictor labels
            dataloader.dataset.Y_dict.update(
                {slice_ind_name: ind_labels, slice_pred_name: pred_labels}
            )
            # Update dataloader task_to_label_dict
            dataloader.task_to_label_dict.update(
                {slice_ind_name: slice_ind_name, slice_pred_name: slice_pred_name}
            )
        msg = (
            f"Loaded slice labels for task {task.name}, slice {slice_name}, "
            f"split {dataloader.split}."
        )
        logger.info(msg)

    return dataloaders


@slicing_function()
def base_slice(example):
    return True
