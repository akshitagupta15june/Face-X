
# system
from __future__ import print_function

# python lib
import importlib

def find_dataloader_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported.
    dataset_filename = "src_common.data." + dataset_name
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataloader = None
    for name, cls in datasetlib.__dict__.items():
        if name == "DataLoader":
            dataloader = cls

    if dataloader is None:
        print("In %s.py, there should be a right class name that matches %s in lowercase." % (
        dataset_filename, "DataLoader"))
        exit(0)

    return dataloader