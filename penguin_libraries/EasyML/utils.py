import contextlib
import copy
import os


def deepcopy_without_verbose(model):
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        return copy.deepcopy(model)
