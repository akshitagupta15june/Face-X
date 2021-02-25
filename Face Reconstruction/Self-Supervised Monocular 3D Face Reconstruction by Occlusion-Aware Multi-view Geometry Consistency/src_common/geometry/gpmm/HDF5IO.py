#!/usr/bin/env python

import h5py
import numpy as np

class HDF5IO:
    def __init__(self, path_file, handler_file = None, mode='a'):
        if(handler_file == None):
            self.handler_file = h5py.File(path_file, mode=mode)
        else:
            self.handler_file = handler_file
    def GetMainKeys(self):
        return self.handler_file.keys()
    def GetValue(self, name):
        return self.handler_file[name]