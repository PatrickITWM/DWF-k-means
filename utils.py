import os
from typing import Any

import dill as dill

from collections import defaultdict


class Saveable:
    """
    Base class to save the full instance with dill.
    """

    def save(self, file_name):
        """
        This method saves the instance of this class or the subclass.

        :param file_name: The path or name where the instance should be stored.
        """
        Saveable.save_object(self, file_name)

    @staticmethod
    def from_file(file_name) -> Any:
        """
        This method loads the classes from the file.

        :param file_name: file name or path of the instance which should be loaded with dill.
        :return: The instance which was stored with dill.
        """
        with open(file_name, 'rb') as file:
            loaded_class = dill.load(file)
        return loaded_class

    @staticmethod
    def save_object(obj, file_name):
        """
        Saves the given object with dill to the given path or file name.

        :param obj: Object to store with dill.
        :param file_name: File name or path to store the object.
        """

        # Create path if necessary
        if not os.path.exists(os.path.dirname(file_name)):
            if not os.path.dirname(file_name) == "":
                try:
                    os.makedirs(os.path.dirname(file_name))
                except OSError as e:  # Guard against race condition
                    print(f"Could not create path {file_name}")
                    raise
        # Save the class
        with open(file_name, 'wb') as file:
            dill.dump(obj, file)


def map_pred_to_true(y_true, y_pred):
    """
    Returns a map to match the labels to the best true label.
    This is necessary, since the labels of the clustering (eg. 0-9)
    doesn't have to match the true labels (eg. A-J) or the label of the clustering doesn't match
    the true label (the clustering label 3 on MNIST data could correspond to the number 5 eg.).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"len y_true: {len(y_true)}, len y_pred: {len(y_pred)}")
    labels = set(y_pred)
    dict_counts = {l: defaultdict(int) for l in labels}
    for i, (label_true, label_pred) in enumerate(zip(y_true, y_pred)):
        dict_counts[label_pred][label_true] += 1
    dict_map = {}
    for label_pred in labels:
        dict_count_dict = dict_counts[label_pred]
        max_occurrence = max(dict_count_dict.values()) if len(dict_count_dict) > 0 else -1
        max_args = [key for key, value in dict_count_dict.items() if value == max_occurrence]
        label_assigend = max_args[0]
        dict_map[label_pred] = label_assigend
    return [dict_map[v] for v in y_pred]
